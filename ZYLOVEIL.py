#!/usr/bin/env python3
from __future__ import annotations
import os, sys, struct, math, hashlib, secrets, threading, lzma, traceback, time, datetime, tempfile, shutil, random, zipfile, io, json
from flask import Flask, request, send_file, render_template_string, jsonify, after_this_request, session, Blueprint

veil_bp = Blueprint('veil', __name__)

PROGRESS = {}
PROGRESS_LOCK = threading.Lock()

# Optional imports
try:
    from PIL import Image, ImageFilter
except Exception:
    Image = None

_HAS_CRYPTOGRAPHY = False
_HAS_PYCRYPTODOME = False
_HAS_ARGON2 = False

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _HAS_CRYPTOGRAPHY = True
except Exception:
    pass

try:
    from Crypto.Cipher import AES as _PyCryptoAES
    _HAS_PYCRYPTODOME = True
except Exception:
    pass

try:
    from argon2.low_level import hash_secret_raw, Type as Argon2Type
    _HAS_ARGON2 = True
except Exception:
    pass

# ---- Constants ----
MAGIC = b"ZYLOSTEG"    # 8 bytes
VERSION = 3  # [UPDATED] Bumped version for new format
ALGO_VERSION = 3  # [FIX 4] Algorithm version for stateless marker
SALT_LEN = 16
IV_LEN = 12
STATELESS_MARKER_LEN = 8  # [FIX 4] Implicit marker length

# Header v3: Magic(8) + Ver(1) + Flags(2) + BPC(1) + Salt(16) + IV(12) + ShareDataLen(8) + 
#            StartTS(8) + EndTS(8) + MaxAtt(1) + AttUsed(1) + ShareIdx(1) + KNeeded(1) + 
#            NTotal(1) + ShareID(4) + MigrationEpoch(4) + Reserved(4)
HEADER_FIXED_LEN = 8 + 1 + 2 + 1 + 16 + 12 + 8 + 8 + 8 + 1 + 1 + 1 + 1 + 1 + 4 + 4 + 4
DEFAULT_BPC_HINT = 2

# Flags
FLAG_TIME_LOCKED = 1 << 0
FLAG_SELF_DESTRUCT = 1 << 1
FLAG_CARRIER_BOUND = 1 << 2
FLAG_NOISE_ADAPTIVE = 1 << 3
FLAG_STATELESS = 1 << 4
FLAG_SEMANTIC_CAMO = 1 << 5
FLAG_POISON_ANTI_FORENSIC = 1 << 6
FLAG_MUTATE_ON_EXTRACT = 1 << 7
FLAG_BITPLANE_MIGRATION = 1 << 8  # [NEW] Adaptive bit-plane migration

# Migration window (6 hours in seconds)
MIGRATION_WINDOW = 6 * 60 * 60

# ---- Progress Tracking ----
def update_progress(task_id: str, stage: str, percent: int, detail: str = ""):
    """Thread-safe progress update"""
    with PROGRESS_LOCK:
        PROGRESS[task_id] = {
            "stage": stage,
            "percent": percent,
            "detail": detail,
            "timestamp": time.time()
        }

def get_progress(task_id: str) -> dict:
    with PROGRESS_LOCK:
        return PROGRESS.get(task_id, {"stage": "idle", "percent": 0})

def clear_progress(task_id: str):
    with PROGRESS_LOCK:
        PROGRESS.pop(task_id, None)

# ---- Shamir Secret Sharing (GF(256)) ----
class GF256:
    """Galois Field 256 arithmetic for Shamir Secret Sharing"""
    def __init__(self):
        self.exp = [0] * 512
        self.log = [0] * 256
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & 0x100: x ^= 0x11D
        self.exp[255] = self.exp[0]
        for i in range(255, 512): 
            self.exp[i] = self.exp[i-255]

    def add(self, a, b): return a ^ b
    def sub(self, a, b): return a ^ b
    def mul(self, a, b):
        if a == 0 or b == 0: return 0
        return self.exp[self.log[a] + self.log[b]]
    def div(self, a, b):
        if b == 0: raise ZeroDivisionError
        if a == 0: return 0
        return self.exp[(self.log[a] - self.log[b]) % 255]

_GF = None
def get_gf():
    global _GF
    if _GF is None:
        _GF = GF256()
    return _GF

def shamir_split(secret: bytes, k: int, n: int) -> list[tuple[int, bytes]]:
    """
    [FIX 1] Split secret into N shares where K are needed for reconstruction.
    Returns list of (share_index, share_bytes).
    """
    if k > n: raise ValueError("K must be <= N")
    if k < 1: raise ValueError("K must be >= 1")
    if n > 255: raise ValueError("N must be <= 255")
    
    gf = get_gf()
    shares = [bytearray() for _ in range(n)]
    
    for byte in secret:
        # Generate k-1 random coefficients for the polynomial
        coeffs = [byte] + [secrets.randbelow(256) for _ in range(k-1)]
        for i in range(n):
            x = i + 1  # x values are 1 to n
            y = 0
            # Evaluate polynomial using Horner's method
            for coeff in reversed(coeffs):
                y = gf.add(gf.mul(y, x), coeff)
            shares[i].append(y)
            
    return [(i+1, bytes(s)) for i, s in enumerate(shares)]

def shamir_recover(shares: list[tuple[int, bytes]], k: int) -> bytes:
    """
    [FIX 1] Recover secret from exactly k shares using Lagrange interpolation.
    """
    if len(shares) < k: 
        raise ValueError(f"Need {k} shares, got {len(shares)}")
    
    gf = get_gf()
    shares = shares[:k]
    xs = [s[0] for s in shares]
    ys_list = [s[1] for s in shares]
    length = len(ys_list[0])
    
    # Verify all shares have same length
    for ys in ys_list:
        if len(ys) != length:
            raise ValueError("Share length mismatch")
    
    secret = bytearray()
    for i in range(length):
        y_vals = [y[i] for y in ys_list]
        # Lagrange Interpolation at x=0
        result = 0
        for j in range(k):
            xj, yj = xs[j], y_vals[j]
            prod = 1
            for m in range(k):
                if j == m: continue
                xm = xs[m]
                # term = xm / (xm - xj)
                nom = xm
                denom = gf.sub(xm, xj)
                if denom == 0:
                    raise ValueError("Duplicate share indices detected")
                term = gf.div(nom, denom)
                prod = gf.mul(prod, term)
            result = gf.add(result, gf.mul(yj, prod))
        secret.append(result)
    return bytes(secret)

# ---- Crypto & Helpers ----

def derive_key(passphrase: str, salt: bytes, carrier_fingerprint: bytes | None = None, length: int = 32) -> bytes:
    """Derive encryption key from passphrase using best available KDF"""
    pw = passphrase.encode("utf-8")
    kdf_salt = salt
    if carrier_fingerprint:
        # Mix carrier fingerprint into salt for carrier binding
        kdf_salt = hashlib.sha256(salt + carrier_fingerprint).digest()[:SALT_LEN]
    
    if _HAS_ARGON2:
        try:
            return hash_secret_raw(
                secret=pw, salt=kdf_salt, 
                time_cost=2, memory_cost=131072, parallelism=2, 
                hash_len=length, type=Argon2Type.ID
            )
        except Exception: pass
    
    if hasattr(hashlib, "scrypt"):
        try:
            return hashlib.scrypt(pw, salt=kdf_salt, n=2**17, r=8, p=1, dklen=length)
        except Exception: pass
        
    return hashlib.pbkdf2_hmac("sha512", pw, kdf_salt, 200_000, dklen=length)

def aead_encrypt(key: bytes, iv: bytes, plaintext: bytes) -> bytes:
    """AEAD encryption using AES-GCM"""
    if _HAS_CRYPTOGRAPHY:
        aesg = AESGCM(key)
        return aesg.encrypt(iv, plaintext, associated_data=None)
    elif _HAS_PYCRYPTODOME:
        cipher = _PyCryptoAES.new(key=key, mode=_PyCryptoAES.MODE_GCM, nonce=iv)
        ct, tag = cipher.encrypt_and_digest(plaintext)
        return ct + tag
    else:
        raise RuntimeError("No crypto library available (install cryptography or pycryptodome)")

def aead_decrypt(key: bytes, iv: bytes, ct: bytes) -> bytes:
    """AEAD decryption using AES-GCM"""
    if _HAS_CRYPTOGRAPHY:
        aesg = AESGCM(key)
        return aesg.decrypt(iv, ct, associated_data=None)
    elif _HAS_PYCRYPTODOME:
        tag = ct[-16:]
        ciphertext = ct[:-16]
        cipher = _PyCryptoAES.new(key=key, mode=_PyCryptoAES.MODE_GCM, nonce=iv)
        return cipher.decrypt_and_verify(ciphertext, tag)
    else:
        raise RuntimeError("No crypto library available")

def prng_bytes(seed: bytes, nbytes: int) -> bytes:
    """Deterministic PRNG using Counter Mode SHA256"""
    out = bytearray()
    ctr = 0
    while len(out) < nbytes:
        out.extend(hashlib.sha256(seed + struct.pack(">Q", ctr)).digest())
        ctr += 1
    return bytes(out[:nbytes])

def deterministic_shuffle(indices: list[int], seed: bytes) -> list[int]:
    """Fisher-Yates shuffle driven by seeded PRNG"""
    arr = indices[:]
    n = len(arr)
    if n <= 1:
        return arr
    rnd = prng_bytes(seed, n * 4 + 32)
    ptr = 0
    for i in range(n - 1, 0, -1):
        if ptr + 4 > len(rnd): break
        val = int.from_bytes(rnd[ptr:ptr+4], "big")
        ptr += 4
        j = val % (i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr

def carrier_fingerprint_from_flat(flat: list[int]) -> bytes:
    """Hash the MSBs of the carrier to bind payload - robust to LSB changes"""
    stride = max(1, len(flat) // 8192)
    b = bytearray()
    mask = 0xF8  # Keep top 5 bits (11111000)
    for i in range(0, len(flat), stride):
        b.append(flat[i] & mask)
    return hashlib.sha256(b).digest()

# ---- [FIX 4] Stateless Mode Hardening ----
def compute_stateless_marker(passphrase: str, carrier_fingerprint: bytes, algo_version: int) -> bytes:
    """
    [FIX 4] Compute a deterministic, keyed implicit marker for stateless mode.
    Marker is derived from passphrase + carrier fingerprint + algorithm version.
    No plaintext magic bytes - cryptographically derived.
    """
    marker_input = (
        passphrase.encode('utf-8') + 
        carrier_fingerprint + 
        struct.pack(">I", algo_version) +
        b"ZYLO_STATELESS_MARKER_V3"
    )
    full_hash = hashlib.sha256(marker_input).digest()
    return full_hash[:STATELESS_MARKER_LEN]

def verify_stateless_marker(extracted_marker: bytes, passphrase: str, 
                            carrier_fingerprint: bytes, algo_version: int) -> bool:
    """
    [FIX 4] Verify the stateless marker matches expected value.
    Uses constant-time comparison to prevent timing attacks.
    """
    expected = compute_stateless_marker(passphrase, carrier_fingerprint, algo_version)
    if len(extracted_marker) != len(expected):
        return False
    result = 0
    for a, b in zip(extracted_marker, expected):
        result |= a ^ b
    return result == 0

# ---- [FIX 5] Adaptive Bit-Plane Migration ----
def compute_migration_epoch(timestamp: float = None) -> int:
    """
    [FEATURE] Compute current migration epoch based on time window.
    Epoch changes every MIGRATION_WINDOW seconds.
    """
    if timestamp is None:
        timestamp = time.time()
    return int(timestamp // MIGRATION_WINDOW)

def derive_bitplane_selection(seed: bytes, epoch: int, pixel_count: int) -> list[int]:
    """
    [FEATURE] Derive which bit-planes to use based on epoch.
    Returns list of bit-plane indices (0-2) for each pixel.
    This causes old copies to fail after migration window.
    """
    epoch_seed = hashlib.sha256(seed + struct.pack(">I", epoch)).digest()
    selections = prng_bytes(epoch_seed, pixel_count)
    # Map to bit-plane: 0 = LSB only, 1 = LSB+1, 2 = LSB+2
    return [(b % 3) for b in selections]

# ---- [FIX 2] Self-Destruct Logic ----
def poison_carrier_data(flat_pixels: list[int], seed: bytes, poison_intensity: float = 1.0) -> list[int]:
    """
    [FIX 2] Irreversibly poison embedded bits using deterministic keyed noise.
    The poisoning is:
    - Irreversible (overwrites actual data)
    - Deterministic (seeded by key material)
    - Not recoverable by brute force (noise pattern unknown without key)
    """
    poison_seed = hashlib.sha256(seed + b"SELF_DESTRUCT_POISON").digest()
    noise = prng_bytes(poison_seed, len(flat_pixels))
    
    poisoned = list(flat_pixels)
    for i in range(len(poisoned)):
        if poison_intensity >= 1.0 or (noise[i] / 255.0) < poison_intensity:
            # XOR lower 3 bits with noise - destroys embedded data
            poisoned[i] = (poisoned[i] & 0xF8) | (noise[i] & 0x07)
    
    return poisoned

def apply_self_destruct(image_path: str, seed: bytes):
    """
    [FIX 2] Apply self-destruct by poisoning the carrier image in-place.
    This modification is permanent and irreversible.
    """
    if not Image:
        raise RuntimeError("Pillow required for self-destruct")
    
    img = Image.open(image_path).convert("RGBA")
    pixels = list(img.getdata())
    flat_pixels = [c for p in pixels for c in p[:3]]
    
    # Poison the data
    poisoned = poison_carrier_data(flat_pixels, seed)
    
    # Reconstruct image
    new_data = []
    for i in range(len(pixels)):
        r = poisoned[i*3]
        g = poisoned[i*3+1]
        b = poisoned[i*3+2]
        a = pixels[i][3]
        new_data.append((r, g, b, a))
    
    out = Image.new("RGBA", img.size)
    out.putdata(new_data)
    out.save(image_path, "PNG")

# ---- [FEATURE] Anti-Forensic Stego Poisoning ----
def generate_camera_noise_profile(seed: bytes, length: int) -> list[int]:
    """
    [FEATURE] Generate noise that mimics camera sensor noise distribution.
    Uses a modified Gaussian-like distribution typical of digital cameras.
    """
    noise_seed = hashlib.sha256(seed + b"CAMERA_NOISE").digest()
    raw_bytes = prng_bytes(noise_seed, length * 2)
    
    noise = []
    for i in range(0, len(raw_bytes) - 1, 2):
        # Box-Muller approximation using two uniform values
        u1 = (raw_bytes[i] + 1) / 256.0
        u2 = (raw_bytes[i+1] + 1) / 256.0
        # Approximate normal distribution, map to 0-7 range (3 LSBs)
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        # Scale and clamp to 0-7
        val = int((z + 4) * 0.875)  # Map roughly -4..4 to 0..7
        val = max(0, min(7, val))
        noise.append(val)
    
    return noise[:length]

def apply_anti_forensic_poisoning(flat_pixels: list[int], capacity_map: list[int], 
                                   seed: bytes, used_indices: set) -> list[int]:
    """
    [FEATURE] Inject decoy LSB noise into unused channels.
    Matches camera-like noise distribution to defeat steganalysis.
    Preserves decode correctness by only modifying unused slots.
    """
    poisoned = list(flat_pixels)
    unused_count = sum(1 for i in range(len(flat_pixels)) if i not in used_indices)
    
    if unused_count == 0:
        return poisoned
    
    noise = generate_camera_noise_profile(seed, unused_count)
    noise_idx = 0
    
    for i in range(len(poisoned)):
        if i not in used_indices:
            cap = capacity_map[i] if i < len(capacity_map) else 1
            if cap > 0 and noise_idx < len(noise):
                # Apply noise to LSBs of unused slot
                mask = (~((1 << cap) - 1)) & 0xFF
                noise_val = noise[noise_idx] & ((1 << cap) - 1)
                poisoned[i] = (poisoned[i] & mask) | noise_val
                noise_idx += 1
    
    return poisoned

# ---- [FEATURE] Stego Mutation Mode ----
def compute_mutation_params(original_seed: bytes, extraction_count: int) -> tuple[bytes, bytes, bytes]:
    """
    [FEATURE] Compute new parameters for mutation after successful extraction.
    Returns (new_permutation_seed, new_salt, new_iv)
    """
    mutation_input = original_seed + struct.pack(">I", extraction_count) + b"MUTATION"
    derived = hashlib.sha512(mutation_input).digest()
    
    new_perm_seed = derived[:32]
    new_salt = derived[32:48]
    new_iv = derived[48:60]
    
    return new_perm_seed, new_salt, new_iv

# ---- Capacity Map Computation ----
def compute_capacity_map(pixels: list[tuple], width: int, height: int, 
                         bpc_hint: int, seed: bytes, flags: int,
                         migration_epoch: int = 0) -> list[int]:
    """
    [FIX 3] Returns list of bits-per-slot for RGB channels.
    Now properly considers all flags including bit-plane migration.
    """
    num_pixels = len(pixels)
    
    use_noise = bool(flags & FLAG_NOISE_ADAPTIVE)
    use_semantic = bool(flags & FLAG_SEMANTIC_CAMO)
    use_migration = bool(flags & FLAG_BITPLANE_MIGRATION)
    
    scores = [0.0] * num_pixels
    
    if use_noise or use_semantic:
        # Mask out LSBs to ensure stability during analysis
        mask = 0xF8 
        lum = [(0.299*(r&mask) + 0.587*(g&mask) + 0.114*(b&mask)) 
               for r, g, b, *_ in pixels]
        w, h = width, height
        
        # Edge detection using cross-difference
        for i in range(num_pixels):
            x, y = i % w, i // w
            l = lum[i]
            diff = 0.0
            count = 0
            if x < w-1: 
                diff += abs(l - lum[i+1])
                count += 1
            if y < h-1:
                diff += abs(l - lum[i+w])
                count += 1
            if x > 0:
                diff += abs(l - lum[i-1])
                count += 1
            if y > 0:
                diff += abs(l - lum[i-w])
                count += 1
            scores[i] = diff / count if count else 0.0

    # Add pseudo-random bias
    bias_bytes = prng_bytes(seed, num_pixels)
    
    # Bit-plane migration adjustment
    migration_adjustment = [0] * num_pixels
    if use_migration and migration_epoch > 0:
        bitplane_selection = derive_bitplane_selection(seed, migration_epoch, num_pixels)
        migration_adjustment = bitplane_selection
    
    final_bpp = []
    
    if not (use_noise or use_semantic):
        for i in range(num_pixels):
            base = bpc_hint
            if use_migration:
                # Shift bit-plane based on epoch
                base = max(1, min(3, base + migration_adjustment[i] - 1))
            final_bpp.append(base)
    else:
        # Normalize scores
        max_s = max(scores) if scores else 1.0
        if max_s < 1e-9:
            max_s = 1.0
        norm_scores = [s / max_s for s in scores]
        
        # Combine with random bias
        combined = []
        for i in range(num_pixels):
            bias = bias_bytes[i] / 255.0
            if use_semantic:
                val = norm_scores[i] * 0.7 + bias * 0.3
            else:
                val = norm_scores[i] * 0.4 + bias * 0.6
            combined.append(val)
        
        # Determine thresholds
        sorted_sc = sorted(combined)
        med = sorted_sc[len(sorted_sc)//2]
        high = sorted_sc[int(len(sorted_sc)*0.85)]
        low = sorted_sc[int(len(sorted_sc)*0.15)]
        
        for i, v in enumerate(combined):
            base_bpc = bpc_hint
            if v > high:
                base_bpc = min(3, bpc_hint + 1)
            elif v < low:
                base_bpc = max(0, bpc_hint - 1)  # [FIX 3] Can be 0 for very smooth areas
            
            # Apply migration adjustment
            if use_migration:
                base_bpc = max(0, min(3, base_bpc + migration_adjustment[i] - 1))
            
            final_bpp.append(base_bpc)

    # [FIX 3] Flatten to RGB slots - each channel gets the pixel's capacity
    bits_map = []
    for bpp in final_bpp:
        bits_map.extend([bpp, bpp, bpp])  # R, G, B
        
    return bits_map

# ---- [FIX 3] Header Embedding with Capacity Awareness ----
def find_valid_header_slots(bits_map: list[int], needed_bits: int, seed: bytes) -> list[int]:
    """
    [FIX 3] Find slots with sufficient capacity for header embedding.
    Returns list of indices that can hold header bits.
    """
    # Find all slots with capacity >= 1
    valid_slots = [i for i, cap in enumerate(bits_map) if cap >= 1]
    
    if len(valid_slots) < needed_bits:
        raise ValueError(f"Insufficient capacity for header. Need {needed_bits} slots, have {len(valid_slots)}")
    
    # Deterministic selection of header slots
    header_seed = hashlib.sha256(seed + b"HEADER_SLOTS").digest()
    shuffled = deterministic_shuffle(valid_slots, header_seed)
    
    return shuffled[:needed_bits]

# ---- Header Packing/Unpacking ----
def pack_header_v3(flags, bpc, salt, iv, share_data_len, start, end, 
                   max_att, att_used, share_idx, k_needed, n_total, share_id, migration_epoch):
    """
    [UPDATED] Pack header v3 with new fields for N total and migration epoch.
    """
    return (
        MAGIC + 
        struct.pack("B", VERSION) +
        struct.pack(">H", flags) + 
        struct.pack("B", bpc) +
        salt +  # 16 bytes
        iv +    # 12 bytes
        struct.pack(">Q", share_data_len) +
        struct.pack(">Q", start) + 
        struct.pack(">Q", end) + 
        struct.pack("B", max_att) + 
        struct.pack("B", att_used) +
        struct.pack("B", share_idx) + 
        struct.pack("B", k_needed) + 
        struct.pack("B", n_total) +
        struct.pack(">I", share_id) +
        struct.pack(">I", migration_epoch) +
        b"\x00" * 4  # Reserved
    )

def unpack_header_v3(b: bytes) -> dict:
    """[UPDATED] Unpack header v3"""
    if len(b) < HEADER_FIXED_LEN:
        raise ValueError(f"Header too short: {len(b)} < {HEADER_FIXED_LEN}")
    
    off = 0
    magic = b[off:off+8]; off += 8
    if magic != MAGIC:
        raise ValueError("Invalid Magic")
    
    ver = b[off]; off += 1
    if ver == 1:
        return unpack_header_v1(b)
    if ver == 2:
        return unpack_header_v2_compat(b)
    if ver != 3:
        raise ValueError(f"Unknown version {ver}")
    
    flags = struct.unpack(">H", b[off:off+2])[0]; off += 2
    bpc = b[off]; off += 1
    salt = b[off:off+16]; off += 16
    iv = b[off:off+12]; off += 12
    share_data_len = struct.unpack(">Q", b[off:off+8])[0]; off += 8
    start = struct.unpack(">Q", b[off:off+8])[0]; off += 8
    end = struct.unpack(">Q", b[off:off+8])[0]; off += 8
    max_att = b[off]; off += 1
    att_used = b[off]; off += 1
    share_idx = b[off]; off += 1
    k_needed = b[off]; off += 1
    n_total = b[off]; off += 1
    share_id = struct.unpack(">I", b[off:off+4])[0]; off += 4
    migration_epoch = struct.unpack(">I", b[off:off+4])[0]; off += 4
    
    return {
        "magic": magic, "ver": ver, "flags": flags, "bpc": bpc,
        "salt": salt, "iv": iv, "share_data_len": share_data_len,
        "start": start, "end": end, "max_att": max_att, "att_used": att_used,
        "share_idx": share_idx, "k_needed": k_needed, "n_total": n_total,
        "share_id": share_id, "migration_epoch": migration_epoch
    }

def unpack_header_v2_compat(b: bytes) -> dict:
    """Compatibility unpacker for v2 headers"""
    off = 9  # Skip magic + version
    flags = struct.unpack(">H", b[off:off+2])[0]; off += 2
    bpc = b[off]; off += 1
    salt = b[off:off+16]; off += 16
    iv = b[off:off+12]; off += 12
    ct_len = struct.unpack(">Q", b[off:off+8])[0]; off += 8
    start = struct.unpack(">Q", b[off:off+8])[0]; off += 8
    end = struct.unpack(">Q", b[off:off+8])[0]; off += 8
    max_att = b[off]; off += 1
    att_used = b[off]; off += 1
    share_idx = b[off]; off += 1
    k_needed = b[off]; off += 1
    share_id = struct.unpack(">I", b[off:off+4])[0]; off += 4
    
    return {
        "magic": MAGIC, "ver": 2, "flags": flags, "bpc": bpc,
        "salt": salt, "iv": iv, "share_data_len": ct_len,
        "start": start, "end": end, "max_att": max_att, "att_used": att_used,
        "share_idx": share_idx, "k_needed": k_needed, "n_total": k_needed,
        "share_id": share_id, "migration_epoch": 0
    }

def unpack_header_v1(b: bytes) -> dict:
    """Compatibility unpacker for v1 headers"""
    off = 9
    flags = b[off]; off += 1
    bpc = b[off]; off += 1
    salt = b[off:off+16]; off += 16
    iv = b[off:off+12]; off += 12
    ct_len = struct.unpack(">Q", b[off:off+8])[0]; off += 8
    start = struct.unpack(">Q", b[off:off+8])[0]; off += 8
    end = struct.unpack(">Q", b[off:off+8])[0]; off += 8
    max_att = b[off]; off += 1
    att_used = b[off]; off += 1
    
    return {
        "magic": MAGIC, "ver": 1, "flags": flags, "bpc": bpc,
        "salt": salt, "iv": iv, "share_data_len": ct_len,
        "start": start, "end": end, "max_att": max_att, "att_used": att_used,
        "share_idx": 1, "k_needed": 1, "n_total": 1,
        "share_id": 0, "migration_epoch": 0
    }

# ---- Bit/Byte Conversion ----
def bits_to_bytes(bits: str) -> bytes:
    """Convert bit string to bytes"""
    if len(bits) % 8 != 0:
        bits = bits.ljust((len(bits)//8 + 1)*8, '0')
    return int(bits, 2).to_bytes(len(bits) // 8, byteorder='big')

def bytes_to_bits(b: bytes) -> str:
    """Convert bytes to bit string"""
    return bin(int.from_bytes(b, byteorder='big'))[2:].zfill(len(b)*8)

# ---- Core Embedding Logic ----

def embed_share(carrier_path: str, output_path: str, share_data: bytes, 
                passphrase: str, flags: int, bpc_hint: int, 
                start_ts: int, end_ts: int, max_att: int,
                share_idx: int, k_needed: int, n_total: int, share_id: int,
                task_id: str = None) -> dict:
    """
    [FIX 1, FIX 3, FEATURES] Embed a single Shamir share into a carrier.
    Now embeds ONLY the share data, not the full payload.
    Returns metadata about the embedding.
    """
    if not Image:
        raise RuntimeError("Pillow library is required")
    
    if task_id:
        update_progress(task_id, "Loading carrier", 10, os.path.basename(carrier_path))
    
    img = Image.open(carrier_path).convert("RGBA")
    pixels = list(img.getdata())
    num_pixels = len(pixels)
    width, height = img.size
    
    # Compute carrier fingerprint
    flat_pixels = [c for p in pixels for c in p[:3]]
    fp = carrier_fingerprint_from_flat(flat_pixels)
    
    if task_id:
        update_progress(task_id, "Deriving keys", 20)
    
    # Compute migration epoch
    migration_epoch = compute_migration_epoch() if (flags & FLAG_BITPLANE_MIGRATION) else 0
    
    used_indices = set()
    
    if flags & FLAG_STATELESS:
        # [FIX 4] Stateless mode with hardened marker
        salt = hashlib.sha256(fp + passphrase.encode()).digest()[:SALT_LEN]
        key = derive_key(passphrase, salt, fp)
        seed_header = hashlib.sha256(key + b"StatelessSeedV3").digest()
        iv = hashlib.sha256(seed_header + b"IV").digest()[:IV_LEN]
        
        # Compute stateless marker
        marker = compute_stateless_marker(passphrase, fp, ALGO_VERSION)
        
        # Encrypt share data
        ciphertext = aead_encrypt(key, iv, share_data)
        
        # Format: Marker(8) + ShareIdx(1) + KNeeded(1) + NTotal(1) + ShareID(4) + CTLen(4) + CT
        data_to_embed = (
            marker +
            struct.pack("B", share_idx) +
            struct.pack("B", k_needed) +
            struct.pack("B", n_total) +
            struct.pack(">I", share_id) +
            struct.pack(">I", len(ciphertext)) +
            ciphertext
        )
        header_len_bits = 0
        header_slots = []
        map_seed = seed_header
    else:
        # Standard mode
        salt = secrets.token_bytes(SALT_LEN)
        iv = secrets.token_bytes(IV_LEN)
        fp_for_key = fp if (flags & FLAG_CARRIER_BOUND) else None
        key = derive_key(passphrase, salt, fp_for_key)
        
        # Encrypt share data
        ciphertext = aead_encrypt(key, iv, share_data)
        
        # Pack header
        header = pack_header_v3(
            flags, bpc_hint, salt, iv, len(ciphertext),
            start_ts, end_ts, max_att, 0,
            share_idx, k_needed, n_total, share_id, migration_epoch
        )
        header_bits = bytes_to_bits(header)
        header_len_bits = len(header_bits)
        data_to_embed = ciphertext
        
        # [FIX] Use deterministic seed for header slots based on passphrase + fingerprint only
        # This allows extraction without knowing the random salt first
        map_seed = hashlib.sha256(passphrase.encode() + fp + b"HEADER_SEED_V3").digest()

    if task_id:
        update_progress(task_id, "Computing capacity map", 30)
    
    # Compute capacity map for header using deterministic seed
    header_bits_map = compute_capacity_map(pixels, width, height, DEFAULT_BPC_HINT, map_seed, 0, 0)
    
    if not (flags & FLAG_STATELESS):
        # [FIX 3] Find header slots using deterministic seed (passphrase + fingerprint based)
        header_slots = find_valid_header_slots(header_bits_map, header_len_bits, map_seed)
    
    # For payload, use salt-based seed for additional security
    if flags & FLAG_STATELESS:
        payload_seed = map_seed
        payload_bits_map = compute_capacity_map(pixels, width, height, bpc_hint, payload_seed, flags, migration_epoch)
    else:
        payload_seed = hashlib.sha256(salt + passphrase.encode() + b"PAYLOAD_V3").digest()
        payload_bits_map = compute_capacity_map(pixels, width, height, bpc_hint, payload_seed, flags, migration_epoch)
    
    # Calculate total capacity
    if flags & FLAG_STATELESS:
        total_capacity = sum(payload_bits_map)
        data_bits = bytes_to_bits(data_to_embed)
        needed = len(data_bits)
    else:
        total_capacity = sum(payload_bits_map)
        data_bits = bytes_to_bits(data_to_embed)
        needed = len(data_bits) + header_len_bits
    
    if needed > total_capacity:
        raise ValueError(
            f"Carrier '{os.path.basename(carrier_path)}' too small. "
            f"Need {needed} bits, have {total_capacity} ({total_capacity//8} bytes)."
        )
    
    if task_id:
        update_progress(task_id, "Embedding data", 50)
    
    indices = list(range(len(flat_pixels)))
    flat_mod = list(flat_pixels)
    
    if flags & FLAG_STATELESS:
        # Stateless: all data via permuted indices
        permuted = deterministic_shuffle(indices, map_seed)
        ptr = 0
        bit_ptr = 0
        
        while bit_ptr < len(data_bits):
            if ptr >= len(permuted):
                raise ValueError("Ran out of embedding slots")
            
            idx = permuted[ptr]
            cap = payload_bits_map[idx]
            
            if cap == 0:
                ptr += 1
                continue
            
            chunk = data_bits[bit_ptr:bit_ptr + cap]
            if len(chunk) < cap:
                chunk = chunk.ljust(cap, '0')
            
            val = flat_mod[idx]
            mask = (~((1 << cap) - 1)) & 0xFF
            flat_mod[idx] = (val & mask) | int(chunk, 2)
            
            used_indices.add(idx)
            bit_ptr += cap
            ptr += 1
    else:
        # [FIX 3] Standard mode: header uses capacity-aware slot selection
        # Write header using capacity-aware slots (1 bit per slot in LSB)
        for h_idx, slot_idx in enumerate(header_slots):
            bit = int(header_bits[h_idx])
            flat_mod[slot_idx] = (flat_mod[slot_idx] & 0xFE) | bit
            used_indices.add(slot_idx)
        
        # Payload: use remaining indices with permutation based on salt
        remaining_indices = [i for i in indices if i not in used_indices]
        permuted = deterministic_shuffle(remaining_indices, payload_seed)
        
        p_ptr = 0  # Data bit pointer
        s_ptr = 0  # Slot pointer
        
        while p_ptr < len(data_bits):
            if s_ptr >= len(permuted):
                raise ValueError("Ran out of embedding slots for payload")
            
            idx = permuted[s_ptr]
            cap = payload_bits_map[idx]
            
            if cap == 0:
                s_ptr += 1
                continue
            
            chunk = data_bits[p_ptr:p_ptr + cap]
            if len(chunk) < cap:
                chunk = chunk.ljust(cap, '0')
            
            val = flat_mod[idx]
            mask = (~((1 << cap) - 1)) & 0xFF
            flat_mod[idx] = (val & mask) | int(chunk, 2)
            
            used_indices.add(idx)
            p_ptr += cap
            s_ptr += 1
    
    if task_id:
        update_progress(task_id, "Applying anti-forensics", 70)
    
    # [FEATURE] Anti-Forensics Poisoning
    if flags & FLAG_POISON_ANTI_FORENSIC:
        poison_seed = hashlib.sha256(map_seed + b"ANTI_FORENSIC").digest()
        flat_mod = apply_anti_forensic_poisoning(flat_mod, payload_bits_map, poison_seed, used_indices)
    
    if task_id:
        update_progress(task_id, "Saving output", 85)
    
    # Reconstruct image
    new_data = []
    for i in range(num_pixels):
        r = flat_mod[i*3]
        g = flat_mod[i*3+1]
        b = flat_mod[i*3+2]
        a = pixels[i][3]
        new_data.append((r, g, b, a))
    
    out = Image.new("RGBA", img.size)
    out.putdata(new_data)
    out.save(output_path, "PNG")
    
    if task_id:
        update_progress(task_id, "Complete", 100)
    
    return {
        "output_path": output_path,
        "share_idx": share_idx,
        "capacity_used": needed,
        "capacity_total": total_capacity,
        "migration_epoch": migration_epoch
    }


def extract_share(stego_path: str, passphrase: str, 
                  task_id: str = None, 
                  allow_mutation: bool = False) -> tuple[dict, bytes, str | None]:
    """
    [FIX 1-4, FEATURES] Extract a single share from a stego carrier.
    Returns (header_dict, share_data, mutated_path_or_none)
    """
    if not Image:
        raise RuntimeError("Pillow library is required")
    
    if task_id:
        update_progress(task_id, "Loading stego image", 10, os.path.basename(stego_path))
    
    img = Image.open(stego_path).convert("RGBA")
    pixels = list(img.getdata())
    flat_pixels = [c for p in pixels for c in p[:3]]
    width, height = img.size
    fp = carrier_fingerprint_from_flat(flat_pixels)
    
    if task_id:
        update_progress(task_id, "Detecting mode", 20)
    
    # [FIX 4] Try stateless detection first with cryptographic marker
    stateless_detected = False
    header = None
    
    # First, attempt standard header extraction using deterministic seed
    # [FIX] Use passphrase + fingerprint based seed (same as embedding)
    try:
        header_seed = hashlib.sha256(passphrase.encode() + fp + b"HEADER_SEED_V3").digest()
        header_bits_map = compute_capacity_map(pixels, width, height, DEFAULT_BPC_HINT, header_seed, 0, 0)
        
        # [FIX 3] Find header slots using same algorithm as embedding
        header_slots = find_valid_header_slots(header_bits_map, HEADER_FIXED_LEN * 8, header_seed)
        
        # Extract header bits from capacity-aware slots
        header_bits = ""
        for slot_idx in header_slots:
            header_bits += str(flat_pixels[slot_idx] & 1)
        
        header_bytes = bits_to_bytes(header_bits)
        header = unpack_header_v3(header_bytes)
        
    except Exception:
        # Header extraction failed - try stateless mode
        stateless_detected = True
    
    if task_id:
        update_progress(task_id, "Extracting payload", 40)
    
    mutated_path = None
    
    if stateless_detected:
        # [FIX 4] Stateless mode with cryptographic marker verification
        salt = hashlib.sha256(fp + passphrase.encode()).digest()[:SALT_LEN]
        key = derive_key(passphrase, salt, fp)
        seed_header = hashlib.sha256(key + b"StatelessSeedV3").digest()
        
        flags_assumed = FLAG_STATELESS | FLAG_SEMANTIC_CAMO | FLAG_CARRIER_BOUND | FLAG_NOISE_ADAPTIVE
        bits_map = compute_capacity_map(pixels, width, height, DEFAULT_BPC_HINT, seed_header, flags_assumed, 0)
        permuted = deterministic_shuffle(list(range(len(flat_pixels))), seed_header)
        
        # Extract: Marker(8) + ShareIdx(1) + KNeeded(1) + NTotal(1) + ShareID(4) + CTLen(4)
        # Total fixed header = 8 + 1 + 1 + 1 + 4 + 4 = 19 bytes = 152 bits
        stateless_header_bits = 19 * 8
        
        extracted_bits = ""
        ptr = 0
        while len(extracted_bits) < stateless_header_bits and ptr < len(permuted):
            idx = permuted[ptr]
            cap = bits_map[idx]
            if cap > 0:
                val = flat_pixels[idx] & ((1 << cap) - 1)
                extracted_bits += bin(val)[2:].zfill(cap)
            ptr += 1
        
        if len(extracted_bits) < stateless_header_bits:
            raise ValueError("Failed to extract stateless header")
        
        header_data = bits_to_bytes(extracted_bits[:stateless_header_bits])
        
        # Verify marker
        extracted_marker = header_data[:STATELESS_MARKER_LEN]
        if not verify_stateless_marker(extracted_marker, passphrase, fp, ALGO_VERSION):
            raise ValueError("Invalid passphrase or corrupted data (marker mismatch)")
        
        # Parse stateless header
        off = STATELESS_MARKER_LEN
        share_idx = header_data[off]; off += 1
        k_needed = header_data[off]; off += 1
        n_total = header_data[off]; off += 1
        share_id = struct.unpack(">I", header_data[off:off+4])[0]; off += 4
        ct_len = struct.unpack(">I", header_data[off:off+4])[0]; off += 4
        
        # Continue extracting ciphertext
        ct_bits_needed = ct_len * 8
        while len(extracted_bits) < stateless_header_bits + ct_bits_needed and ptr < len(permuted):
            idx = permuted[ptr]
            cap = bits_map[idx]
            if cap > 0:
                val = flat_pixels[idx] & ((1 << cap) - 1)
                extracted_bits += bin(val)[2:].zfill(cap)
            ptr += 1
        
        ct_bits = extracted_bits[stateless_header_bits:stateless_header_bits + ct_bits_needed]
        ciphertext = bits_to_bytes(ct_bits)
        
        iv = hashlib.sha256(seed_header + b"IV").digest()[:IV_LEN]
        
        try:
            share_data = aead_decrypt(key, iv, ciphertext)
        except Exception as e:
            raise ValueError(f"Decryption failed: wrong passphrase or corrupted data")
        
        header = {
            "flags": flags_assumed,
            "share_idx": share_idx,
            "k_needed": k_needed,
            "n_total": n_total,
            "share_id": share_id,
            "bpc": DEFAULT_BPC_HINT,
            "start": 0,
            "end": 0,
            "max_att": 0,
            "att_used": 0,
            "migration_epoch": 0
        }
        
    else:
        # Standard mode extraction
        flags = header["flags"]
        
        # [FIX 2] Check and handle self-destruct
        if flags & FLAG_SELF_DESTRUCT:
            if header["att_used"] >= header["max_att"] and header["max_att"] > 0:
                raise ValueError("DESTROYED: Maximum extraction attempts exceeded. Data is irrecoverable.")
        
        # Time lock check
        if flags & FLAG_TIME_LOCKED:
            now = time.time()
            if header["start"] > 0 and now < header["start"]:
                raise ValueError(f"Time lock active: cannot access until {datetime.datetime.fromtimestamp(header['start'])}")
            if header["end"] > 0 and now > header["end"]:
                raise ValueError(f"Time lock expired: access window ended at {datetime.datetime.fromtimestamp(header['end'])}")
        
        # [FEATURE] Bit-plane migration check
        if flags & FLAG_BITPLANE_MIGRATION:
            current_epoch = compute_migration_epoch()
            if header["migration_epoch"] != current_epoch:
                raise ValueError(
                    f"Bit-plane migration: this carrier was created in epoch {header['migration_epoch']}, "
                    f"current epoch is {current_epoch}. Data is no longer accessible from this copy."
                )
        
        salt = header["salt"]
        iv = header["iv"]
        fp_for_key = fp if (flags & FLAG_CARRIER_BOUND) else None
        key = derive_key(passphrase, salt, fp_for_key)
        
        # [FIX] Use salt-based seed for payload (same as embedding)
        payload_seed = hashlib.sha256(salt + passphrase.encode() + b"PAYLOAD_V3").digest()
        payload_bits_map = compute_capacity_map(pixels, width, height, header["bpc"], payload_seed, flags, header["migration_epoch"])
        
        # [FIX 3] Get header slots to exclude them (using deterministic header seed)
        header_seed = hashlib.sha256(passphrase.encode() + fp + b"HEADER_SEED_V3").digest()
        header_bits_map = compute_capacity_map(pixels, width, height, DEFAULT_BPC_HINT, header_seed, 0, 0)
        header_slots_set = set(find_valid_header_slots(header_bits_map, HEADER_FIXED_LEN * 8, header_seed))
        
        # Extract payload from remaining slots
        remaining_indices = [i for i in range(len(flat_pixels)) if i not in header_slots_set]
        permuted = deterministic_shuffle(remaining_indices, payload_seed)
        
        needed_bits = header["share_data_len"] * 8
        collected_bits = ""
        
        for idx in permuted:
            if len(collected_bits) >= needed_bits:
                break
            cap = payload_bits_map[idx]
            if cap == 0:
                continue
            val = flat_pixels[idx] & ((1 << cap) - 1)
            collected_bits += bin(val)[2:].zfill(cap)
        
        if len(collected_bits) < needed_bits:
            raise ValueError("Failed to extract enough payload bits")
        
        ciphertext = bits_to_bytes(collected_bits[:needed_bits])
        
        try:
            share_data = aead_decrypt(key, iv, ciphertext)
        except Exception as e:
            # [FIX 2] Self-destruct on failed decryption
            if flags & FLAG_SELF_DESTRUCT:
                new_att_used = header["att_used"] + 1
                
                if new_att_used >= header["max_att"] and header["max_att"] > 0:
                    # Trigger destruction
                    destroy_seed = hashlib.sha256(salt + iv + b"DESTROY").digest()
                    apply_self_destruct(stego_path, destroy_seed)
                    raise ValueError(
                        "SELF-DESTRUCT TRIGGERED: Maximum attempts exceeded. "
                        "Carrier has been permanently destroyed."
                    )
                else:
                    # Update attempt counter in the image
                    update_attempt_counter(stego_path, header, new_att_used, header_seed, header_bits_map)
                    raise ValueError(
                        f"Decryption failed. Attempt {new_att_used}/{header['max_att']}. "
                        f"WARNING: {header['max_att'] - new_att_used} attempts remaining before self-destruct."
                    )
            else:
                raise ValueError("Decryption failed: wrong passphrase or corrupted data")
    
    if task_id:
        update_progress(task_id, "Post-processing", 80)
    
    # [FEATURE] Stego Mutation Mode
    if header.get("flags", 0) & FLAG_MUTATE_ON_EXTRACT and allow_mutation:
        mutated_path = perform_mutation(stego_path, share_data, passphrase, header)
    
    if task_id:
        update_progress(task_id, "Complete", 100)
    
    return header, share_data, mutated_path


def update_attempt_counter(stego_path: str, header: dict, new_att_used: int, 
                           seed: bytes, bits_map: list[int]):
    """
    [FIX 2] Update the attempt counter in the stego image header.
    """
    img = Image.open(stego_path).convert("RGBA")
    pixels = list(img.getdata())
    flat_pixels = [c for p in pixels for c in p[:3]]
    
    # Rebuild header with updated attempt count
    new_header = pack_header_v3(
        header["flags"], header["bpc"], header["salt"], header["iv"],
        header["share_data_len"], header["start"], header["end"],
        header["max_att"], new_att_used,
        header["share_idx"], header["k_needed"], header["n_total"],
        header["share_id"], header["migration_epoch"]
    )
    new_header_bits = bytes_to_bits(new_header)
    
    # [FIX 3] Get header slots
    header_slots = find_valid_header_slots(bits_map, len(new_header_bits), seed)
    
    # Update header bits
    flat_mod = list(flat_pixels)
    for h_idx, slot_idx in enumerate(header_slots):
        bit = int(new_header_bits[h_idx])
        flat_mod[slot_idx] = (flat_mod[slot_idx] & 0xFE) | bit
    
    # Save updated image
    new_data = []
    for i in range(len(pixels)):
        r = flat_mod[i*3]
        g = flat_mod[i*3+1]
        b = flat_mod[i*3+2]
        a = pixels[i][3]
        new_data.append((r, g, b, a))
    
    out = Image.new("RGBA", img.size)
    out.putdata(new_data)
    out.save(stego_path, "PNG")


def perform_mutation(stego_path: str, share_data: bytes, passphrase: str, 
                     header: dict) -> str:
    """
    [FEATURE] Re-embed payload with rotated parameters after successful extraction.
    Returns path to mutated carrier.
    """
    # Generate new mutation parameters
    original_seed = header.get("salt", b"") + header.get("iv", b"")
    extraction_count = header.get("att_used", 0) + 1
    new_perm_seed, new_salt, new_iv = compute_mutation_params(original_seed, extraction_count)
    
    # Create output path
    base, ext = os.path.splitext(stego_path)
    mutated_path = f"{base}_mutated{ext}"
    
    # Read original image for re-embedding
    img = Image.open(stego_path).convert("RGBA")
    
    # Save as temp file for re-embedding
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name, "PNG")
        temp_carrier = tmp.name
    
    try:
        # Re-embed with new parameters
        flags = header.get("flags", 0)
        # Remove mutation flag to prevent infinite mutation
        flags &= ~FLAG_MUTATE_ON_EXTRACT
        
        embed_share(
            temp_carrier, mutated_path, share_data, passphrase, flags,
            header.get("bpc", DEFAULT_BPC_HINT),
            header.get("start", 0), header.get("end", 0),
            header.get("max_att", 0),
            header.get("share_idx", 1), header.get("k_needed", 1),
            header.get("n_total", 1), header.get("share_id", 0)
        )
    finally:
        os.unlink(temp_carrier)
    
    return mutated_path


# ---- Main Orchestration ----

def embed_master(secret_file: str, carrier_files: list[str], passphrase: str, 
                 flags: int, k_needed: int, start_ts: int, end_ts: int, 
                 max_att: int, task_id: str = None) -> list[str]:
    """
    [FIX 1] Master embedding function - TRUE SHAMIR IMPLEMENTATION
    
    The payload is encrypted, then the CIPHERTEXT is split into N Shamir shares.
    Each carrier embeds ONLY ONE share of the ciphertext.
    A single carrier is cryptographically useless - you need K carriers to 
    reconstruct the ciphertext, then decrypt.
    """
    with open(secret_file, "rb") as f:
        data = f.read()
    
    if task_id:
        update_progress(task_id, "Compressing payload", 5)
    
    # [FIX] Include filename in compressed payload to preserve extension
    original_filename = os.path.basename(secret_file)
    filename_bytes = original_filename.encode("utf-8")
    blob_to_compress = struct.pack(">H", len(filename_bytes)) + filename_bytes + data
    
    # Compress the payload
    payload = lzma.compress(blob_to_compress, preset=6)
    
    if task_id:
        update_progress(task_id, "Encrypting payload", 10)
    
    # Generate master encryption key and IV
    master_key = secrets.token_bytes(32)
    master_iv = secrets.token_bytes(IV_LEN)
    
    # Encrypt the compressed payload
    encrypted_payload = aead_encrypt(master_key, master_iv, payload)
    
    # [FIX 1] Combine key and IV with encrypted payload for splitting
    # Format: MasterKey(32) + MasterIV(12) + EncryptedPayload
    full_blob = master_key + master_iv + encrypted_payload
    
    n = len(carrier_files)
    if k_needed > n:
        k_needed = n
    if k_needed < 1:
        k_needed = 1
    
    if task_id:
        update_progress(task_id, f"Splitting into {n} shares (K={k_needed})", 15)
    
    # [FIX 1] Split the ENTIRE blob (key + iv + ciphertext) using Shamir
    # This ensures no single carrier has any useful data
    shares = shamir_split(full_blob, k_needed, n)
    
    # Generate unique share ID for this set
    share_id = secrets.randbits(32)
    
    results = []
    
    for i, carrier in enumerate(carrier_files):
        if task_id:
            progress = 20 + int((70 * i) / n)
            update_progress(task_id, f"Embedding share {i+1}/{n}", progress, os.path.basename(carrier))
        
        share_idx, share_data = shares[i]
        
        # Output filename
        out_name = f"stego_{i+1}of{n}_{os.path.basename(carrier)}"
        out_path = os.path.join(os.path.dirname(secret_file) or ".", out_name)
        
        embed_share(
            carrier, out_path, share_data, passphrase, flags,
            DEFAULT_BPC_HINT, start_ts, end_ts, max_att,
            share_idx, k_needed, n, share_id
        )
        
        results.append(out_path)
    
    if task_id:
        update_progress(task_id, "Complete", 100)
    
    return results


def extract_master(stego_files: list[str], passphrase: str, 
                   task_id: str = None, allow_mutation: bool = False) -> tuple[bytes, str, list[str]]:
    """
    [FIX 1] Master extraction function - TRUE SHAMIR RECONSTRUCTION
    
    Collects shares from K carriers, reconstructs the ciphertext using 
    Shamir recovery, then decrypts to get the original payload.
    Returns (decrypted_data, filename, list_of_mutated_paths)
    """
    shares = []
    headers = []
    mutated_paths = []
    share_id_seen = None
    k_needed = None
    n_total = None
    
    if task_id:
        update_progress(task_id, "Processing carriers", 10)
    
    for i, path in enumerate(stego_files):
        if task_id:
            progress = 10 + int((40 * i) / len(stego_files))
            update_progress(task_id, f"Extracting share {i+1}/{len(stego_files)}", progress, os.path.basename(path))
        
        try:
            head, share_data, mutated = extract_share(path, passphrase, allow_mutation=allow_mutation)
            
            # Validate share belongs to same set
            if share_id_seen is None:
                share_id_seen = head.get("share_id", 0)
                k_needed = head.get("k_needed", 1)
                n_total = head.get("n_total", 1)
            elif head.get("share_id", 0) != share_id_seen:
                # Different share set - skip
                continue
            
            shares.append((head["share_idx"], share_data))
            headers.append(head)
            
            if mutated:
                mutated_paths.append(mutated)
                
        except Exception as e:
            # Log but continue trying other files
            print(f"Warning: Failed to extract from {path}: {e}")
            continue
    
    if not shares:
        raise ValueError("No valid shares could be extracted from provided carriers")
    
    if k_needed is None:
        k_needed = 1
    
    if len(shares) < k_needed:
        raise ValueError(
            f"Insufficient shares: need {k_needed}, but only extracted {len(shares)}. "
            f"Provide at least {k_needed} valid carrier images."
        )
    
    if task_id:
        update_progress(task_id, "Reconstructing secret", 60)
    
    # [FIX 1] Recover the full blob using Shamir
    full_blob = shamir_recover(shares, k_needed)
    
    # Parse the blob: MasterKey(32) + MasterIV(12) + EncryptedPayload
    if len(full_blob) < 32 + IV_LEN:
        raise ValueError("Reconstructed data is too short - corruption or wrong shares")
    
    master_key = full_blob[:32]
    master_iv = full_blob[32:32+IV_LEN]
    encrypted_payload = full_blob[32+IV_LEN:]
    
    if task_id:
        update_progress(task_id, "Decrypting payload", 75)
    
    # Decrypt
    try:
        compressed_payload = aead_decrypt(master_key, master_iv, encrypted_payload)
    except Exception as e:
        raise ValueError(f"Decryption failed - shares may be corrupted or from different sets: {e}")
    
    if task_id:
        update_progress(task_id, "Decompressing", 90)
    
    # Decompress
    try:
        decompressed_blob = lzma.decompress(compressed_payload)
        # [FIX] Extract filename from blob
        fname_len = struct.unpack(">H", decompressed_blob[:2])[0]
        filename = decompressed_blob[2:2+fname_len].decode("utf-8")
        data = decompressed_blob[2+fname_len:]
    except Exception as e:
        # Fallback for older versions that didn't include filename metadata
        try:
            data = lzma.decompress(compressed_payload)
            filename = "extracted_secret"
        except Exception:
            raise ValueError(f"Decompression failed - data may be corrupted: {e}")
    
    if task_id:
        update_progress(task_id, "Complete", 100)
    
    return data, filename, mutated_paths


# ---- Flask Application ----

# app = Flask(__name__)  <-- Removed global app for Blueprint support
# app.secret_key = secrets.token_hex(32) <-- Managed by main app

@veil_bp.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@veil_bp.route('/progress/<task_id>')
def progress_endpoint(task_id):
    """Get progress for a running task"""
    return jsonify(get_progress(task_id))

@veil_bp.route('/embed', methods=['POST'])
def handle_embed():
    """Handle embedding request"""
    tmp = tempfile.mkdtemp()
    task_id = request.form.get('task_id', secrets.token_hex(8))
    
    try:
        update_progress(task_id, "Initializing", 0)
        
        secret = request.files.get('secret')
        carriers = request.files.getlist('carrier')
        passphrase = request.form.get('passphrase', '')
        
        if not secret:
            return jsonify({"error": "No secret file provided"}), 400
        if not carriers:
            return jsonify({"error": "No carrier images provided"}), 400
        if not passphrase:
            return jsonify({"error": "Passphrase is required"}), 400
        if len(passphrase) < 8:
            return jsonify({"error": "Passphrase must be at least 8 characters"}), 400
        
        # Parse flags
        flags = 0
        if request.form.get('flag_time'): flags |= FLAG_TIME_LOCKED
        if request.form.get('flag_self_destruct'): flags |= FLAG_SELF_DESTRUCT
        if request.form.get('flag_carrier'): flags |= FLAG_CARRIER_BOUND
        if request.form.get('flag_noise'): flags |= FLAG_NOISE_ADAPTIVE
        if request.form.get('flag_stateless'): flags |= FLAG_STATELESS
        if request.form.get('flag_semantic'): flags |= FLAG_SEMANTIC_CAMO
        if request.form.get('flag_poison'): flags |= FLAG_POISON_ANTI_FORENSIC
        if request.form.get('flag_mutate'): flags |= FLAG_MUTATE_ON_EXTRACT
        if request.form.get('flag_migration'): flags |= FLAG_BITPLANE_MIGRATION
        
        # Parse time locks
        start_ts, end_ts = 0, 0
        if request.form.get('start_date'):
            try:
                start_ts = int(datetime.datetime.strptime(
                    request.form['start_date'], "%Y-%m-%d"
                ).timestamp())
            except: pass
        if request.form.get('end_date'):
            try:
                end_ts = int(datetime.datetime.strptime(
                    request.form['end_date'], "%Y-%m-%d"
                ).timestamp())
            except: pass
        
        k_needed = int(request.form.get('k_needed', 1))
        max_att = int(request.form.get('max_attempts', 3)) if (flags & FLAG_SELF_DESTRUCT) else 0
        
        update_progress(task_id, "Uploading files", 2)
        
        # Save uploaded files
        secret_filename = os.path.basename(secret.filename) or "secret"
        s_path = os.path.join(tmp, secret_filename)
        secret.save(s_path)
        
        c_paths = []
        for c in carriers:
            # Sanitize filename
            safe_name = os.path.basename(c.filename).replace('..', '_')
            p = os.path.join(tmp, safe_name)
            c.save(p)
            c_paths.append(p)
        
        update_progress(task_id, "Starting embedding", 5)
        
        # Perform embedding
        out_files = embed_master(
            s_path, c_paths, passphrase, flags, k_needed, 
            start_ts, end_ts, max_att, task_id
        )
        
        update_progress(task_id, "Creating archive", 95)
        
        # Create ZIP archive
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in out_files:
                zf.write(f, os.path.basename(f))
        mem.seek(0)
        
        update_progress(task_id, "Complete", 100)
        clear_progress(task_id)
        
        return send_file(
            mem, as_attachment=True, 
            download_name="stego_shares.zip", 
            mimetype="application/zip"
        )
        
    except Exception as e:
        traceback.print_exc()
        clear_progress(task_id)
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@veil_bp.route('/extract', methods=['POST'])
def handle_extract():
    """Handle extraction request"""
    tmp = tempfile.mkdtemp()
    task_id = request.form.get('task_id', secrets.token_hex(8))
    
    try:
        update_progress(task_id, "Initializing", 0)
        
        stegos = request.files.getlist('stego')
        passphrase = request.form.get('passphrase', '')
        allow_mutation = request.form.get('allow_mutation') == 'on'
        
        if not stegos:
            return jsonify({"error": "No stego images provided"}), 400
        if not passphrase:
            return jsonify({"error": "Passphrase is required"}), 400
        
        update_progress(task_id, "Uploading files", 2)
        
        # Save uploaded files
        paths = []
        for s in stegos:
            safe_name = os.path.basename(s.filename).replace('..', '_')
            p = os.path.join(tmp, safe_name)
            s.save(p)
            paths.append(p)
        
        update_progress(task_id, "Starting extraction", 5)
        
        # Perform extraction
        data, filename, mutated_paths = extract_master(paths, passphrase, task_id, allow_mutation)
        
        update_progress(task_id, "Preparing download", 95)
        clear_progress(task_id)
        
        # If mutation occurred, include mutated files
        if mutated_paths:
            mem = io.BytesIO()
            with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(filename, data)
                for mp in mutated_paths:
                    if os.path.exists(mp):
                        zf.write(mp, f"mutated/{os.path.basename(mp)}")
            mem.seek(0)
            return send_file(
                mem, as_attachment=True,
                download_name="extraction_result.zip",
                mimetype="application/zip"
            )
        else:
            return send_file(
                io.BytesIO(data), as_attachment=True,
                download_name=filename
            )
        
    except Exception as e:
        traceback.print_exc()
        clear_progress(task_id)
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- Premium Glass Morphism UI Template ----

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZYLO VEIL v3 — Advanced Steganography</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.1);
            --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            --accent-purple: #a855f7;
            --accent-pink: #ec4899;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-orange: #f97316;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            min-height: 100vh;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            background-attachment: fixed;
            color: var(--text-primary);
            padding: 2rem;
            overflow-x: hidden;
        }
        
        /* Animated Background */
        .bg-animation {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            overflow: hidden;
        }
        
        .bg-animation::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(168, 85, 247, 0.1) 0%, transparent 50%);
            animation: rotate 30s linear infinite;
        }
        
        .bg-animation::after {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(236, 72, 153, 0.08) 0%, transparent 40%);
            animation: rotate 25s linear infinite reverse;
        }
        
        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        /* Glass Card */
        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1px solid var(--glass-border);
            box-shadow: var(--glass-shadow);
            padding: 2rem;
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
            border-color: rgba(255, 255, 255, 0.15);
        }
        
        /* Header */
        .header {
            text-align: center;
            margin-bottom: 3rem;
        }
        
        .logo {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-pink), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
            letter-spacing: -1px;
        }
        
        .subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 300;
        }
        
        .version-badge {
            display: inline-block;
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-pink));
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }
        
        /* Main Grid */
        .main-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        /* Section Headers */
        .section-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }
        
        .section-header i {
            font-size: 1.5rem;
        }
        
        .section-header h2 {
            font-size: 1.5rem;
            font-weight: 600;
        }
        
        .embed-header i { color: var(--accent-purple); }
        .extract-header i { color: var(--accent-green); }
        
        /* Form Elements */
        .form-group {
            margin-bottom: 1.25rem;
        }
        
        .form-label {
            display: block;
            margin-bottom: 0.5rem;
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-secondary);
        }
        
        .form-input {
            width: 100%;
            padding: 0.875rem 1rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            color: var(--text-primary);
            font-size: 0.95rem;
            transition: all 0.2s ease;
        }
        
        .form-input:focus {
            outline: none;
            border-color: var(--accent-purple);
            box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.2);
        }
        
        .form-input::placeholder {
            color: rgba(255, 255, 255, 0.3);
        }
        
        .form-input[type="file"] {
            padding: 0.75rem;
        }
        
        .form-input[type="file"]::file-selector-button {
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-pink));
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            cursor: pointer;
            margin-right: 1rem;
            transition: opacity 0.2s;
        }
        
        .form-input[type="file"]::file-selector-button:hover {
            opacity: 0.9;
        }
        
        /* Flags Grid */
        .flags-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
            background: rgba(0, 0, 0, 0.2);
            padding: 1.25rem;
            border-radius: 16px;
            margin-bottom: 1.25rem;
        }
        
        .flag-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem;
            border-radius: 8px;
            transition: background 0.2s;
            cursor: pointer;
        }
        
        .flag-item:hover {
            background: rgba(255, 255, 255, 0.05);
        }
        
        .flag-item input[type="checkbox"] {
            width: 18px;
            height: 18px;
            accent-color: var(--accent-purple);
            cursor: pointer;
        }
        
        .flag-item label {
            font-size: 0.85rem;
            cursor: pointer;
            user-select: none;
        }
        
        .flag-item.destructive label {
            color: var(--accent-red);
        }
        
        .flag-item.destructive input {
            accent-color: var(--accent-red);
        }
        
        /* Warning Banner */
        .warning-banner {
            display: none;
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(249, 115, 22, 0.2));
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1.25rem;
            animation: pulse-warning 2s ease-in-out infinite;
        }
        
        .warning-banner.active {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .warning-banner i {
            color: var(--accent-red);
            font-size: 1.5rem;
        }
        
        .warning-banner p {
            font-size: 0.85rem;
            color: #fca5a5;
            line-height: 1.4;
        }
        
        @keyframes pulse-warning {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        /* Options Row */
        .options-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-bottom: 1.25rem;
        }
        
        .option-group {
            display: flex;
            flex-direction: column;
        }
        
        .option-group label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-bottom: 0.35rem;
        }
        
        .option-group input {
            padding: 0.6rem 0.75rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            color: var(--text-primary);
            font-size: 0.9rem;
        }
        
        .option-group input:focus {
            outline: none;
            border-color: var(--accent-purple);
        }
        
        /* Submit Buttons */
        .submit-btn {
            width: 100%;
            padding: 1rem 1.5rem;
            border: none;
            border-radius: 14px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
        }
        
        .embed-btn {
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-pink));
            color: white;
        }
        
        .extract-btn {
            background: linear-gradient(135deg, var(--accent-green), var(--accent-blue));
            color: white;
        }
        
        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(168, 85, 247, 0.4);
        }
        
        .submit-btn:active {
            transform: translateY(0);
        }
        
        /* Progress Overlay */
        .progress-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        
        .progress-overlay.active {
            display: flex;
        }
        
        .progress-card {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 3rem;
            text-align: center;
            min-width: 400px;
        }
        
        .progress-spinner {
            width: 80px;
            height: 80px;
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top-color: var(--accent-purple);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1.5rem;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .progress-stage {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .progress-detail {
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
        }
        
        .progress-bar-container {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-purple), var(--accent-pink));
            border-radius: 4px;
            transition: width 0.3s ease;
            width: 0%;
        }
        
        .progress-percent {
            margin-top: 0.75rem;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }
        
        /* Drop Zone */
        .drop-zone {
            border: 2px dashed rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            padding: 2.5rem;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
        }
        
        .drop-zone:hover, .drop-zone.dragover {
            border-color: var(--accent-green);
            background: rgba(16, 185, 129, 0.1);
        }
        
        .drop-zone i {
            font-size: 3rem;
            color: var(--text-secondary);
            margin-bottom: 1rem;
        }
        
        .drop-zone p {
            color: var(--text-secondary);
        }
        
        .drop-zone input[type="file"] {
            position: absolute;
            inset: 0;
            opacity: 0;
            cursor: pointer;
        }
        
        /* Features List */
        .features-list {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-top: 3rem;
            max-width: 1400px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .feature-badge {
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            transition: all 0.2s ease;
        }
        
        .feature-badge:hover {
            transform: translateY(-2px);
            border-color: var(--accent-purple);
        }
        
        .feature-badge i {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-pink));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .feature-badge span {
            display: block;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }
        
        /* Responsive */
        @media (max-width: 1024px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
            .features-list {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        
        @media (max-width: 600px) {
            body {
                padding: 1rem;
            }
            .glass-card {
                padding: 1.5rem;
            }
            .flags-grid {
                grid-template-columns: 1fr;
            }
            .options-row {
                grid-template-columns: 1fr;
            }
            .features-list {
                grid-template-columns: 1fr;
            }
        }
        
        /* Toast Notifications */
        .toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            padding: 1rem 1.5rem;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            transform: translateX(150%);
            transition: transform 0.3s ease;
            z-index: 1001;
        }
        
        .toast.show {
            transform: translateX(0);
        }
        
        .toast.success { border-color: var(--accent-green); }
        .toast.error { border-color: var(--accent-red); }
        
        .toast i.fa-check-circle { color: var(--accent-green); }
        .toast i.fa-exclamation-circle { color: var(--accent-red); }
    </style>
</head>
<body>
    <div class="bg-animation"></div>
    
    <header class="header">
        <h1 class="logo">ZYLO VEIL</h1>
        <p class="subtitle">Advanced Multi-Carrier Steganography System</p>
        <span class="version-badge">v3.0 — True Shamir Split</span>
    </header>
    
    <main class="main-grid">
        <!-- Embed Panel -->
        <section class="glass-card">
            <div class="section-header embed-header">
                <i class="fas fa-shield-halved"></i>
                <h2>Embed (Veil)</h2>
            </div>
            
            <form id="embedForm" action="/embed" method="post" enctype="multipart/form-data">
                <input type="hidden" name="task_id" id="embedTaskId">
                <div class="form-group">
                    <label class="form-label">Secret File</label>
                    <input type="file" name="secret" required class="form-input">
                </div>
                
                <div class="form-group">
                    <label class="form-label">Carrier Images (PNG)</label>
                    <input type="file" name="carrier" multiple accept=".png" required class="form-input">
                </div>
                
                <div class="form-group">
                    <label class="form-label">Passphrase (min 8 characters)</label>
                    <input type="password" name="passphrase" required minlength="8" 
                           class="form-input" placeholder="Enter a strong passphrase">
                </div>
                
                <div class="flags-grid">
                    <div class="flag-item">
                        <input type="checkbox" name="flag_carrier" id="f_carrier" checked>
                        <label for="f_carrier"><i class="fas fa-fingerprint"></i> Carrier Bound</label>
                    </div>
                    <div class="flag-item">
                        <input type="checkbox" name="flag_noise" id="f_noise" checked>
                        <label for="f_noise"><i class="fas fa-wave-square"></i> Noise Adaptive</label>
                    </div>
                    <div class="flag-item">
                        <input type="checkbox" name="flag_semantic" id="f_semantic" checked>
                        <label for="f_semantic"><i class="fas fa-brain"></i> Semantic Camo</label>
                    </div>
                    <div class="flag-item">
                        <input type="checkbox" name="flag_poison" id="f_poison">
                        <label for="f_poison"><i class="fas fa-ghost"></i> Anti-Forensics</label>
                    </div>
                    <div class="flag-item">
                        <input type="checkbox" name="flag_time" id="f_time">
                        <label for="f_time"><i class="fas fa-clock"></i> Time Lock</label>
                    </div>
                    <div class="flag-item">
                        <input type="checkbox" name="flag_stateless" id="f_stateless">
                        <label for="f_stateless"><i class="fas fa-eye-slash"></i> Stateless Mode</label>
                    </div>
                    <div class="flag-item">
                        <input type="checkbox" name="flag_mutate" id="f_mutate">
                        <label for="f_mutate"><i class="fas fa-dna"></i> Mutation Mode</label>
                    </div>
                    <div class="flag-item">
                        <input type="checkbox" name="flag_migration" id="f_migration">
                        <label for="f_migration"><i class="fas fa-random"></i> Bit Migration</label>
                    </div>
                    <div class="flag-item destructive">
                        <input type="checkbox" name="flag_self_destruct" id="f_destruct">
                        <label for="f_destruct"><i class="fas fa-bomb"></i> Self-Destruct</label>
                    </div>
                </div>
                
                <div class="warning-banner" id="destructWarning">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p><strong>⚠️ Self-Destruct Enabled:</strong> After the configured number of failed extraction attempts, the carrier will be permanently destroyed and data will be irrecoverable.</p>
                </div>
                
                <div class="options-row">
                    <div class="option-group">
                        <label>Required Shares (K)</label>
                        <input type="number" name="k_needed" value="2" min="1" max="255">
                    </div>
                    <div class="option-group">
                        <label>Max Attempts</label>
                        <input type="number" name="max_attempts" value="3" min="1" max="255">
                    </div>
                    <div class="option-group">
                        <label>Start Date</label>
                        <input type="date" name="start_date">
                    </div>
                </div>
                
                <div class="options-row" style="grid-template-columns: 1fr 1fr;">
                    <div class="option-group">
                        <label>End Date</label>
                        <input type="date" name="end_date">
                    </div>
                    <div></div>
                </div>
                
                <button type="submit" class="submit-btn embed-btn">
                    <i class="fas fa-lock"></i>
                    Embed & Protect
                </button>
            </form>
        </section>
        
        <!-- Extract Panel -->
        <section class="glass-card">
            <div class="section-header extract-header">
                <i class="fas fa-unlock-keyhole"></i>
                <h2>Extract (Unveil)</h2>
            </div>
            
            <form id="extractForm" action="/extract" method="post" enctype="multipart/form-data">
                <input type="hidden" name="task_id" id="extractTaskId">
                <div class="form-group">
                    <div class="drop-zone" id="dropZone">
                        <input type="file" name="stego" multiple accept=".png" required>
                        <i class="fas fa-cloud-upload-alt"></i>
                        <p>Drop stego carrier images here<br><small>or click to browse</small></p>
                    </div>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Passphrase</label>
                    <input type="password" name="passphrase" required 
                           class="form-input" placeholder="Enter the passphrase used during embedding">
                </div>
                
                <div class="flags-grid" style="grid-template-columns: 1fr;">
                    <div class="flag-item">
                        <input type="checkbox" name="allow_mutation" id="e_mutate">
                        <label for="e_mutate"><i class="fas fa-dna"></i> Allow Mutation (invalidates extracted carriers)</label>
                    </div>
                </div>
                
                <button type="submit" class="submit-btn extract-btn">
                    <i class="fas fa-key"></i>
                    Extract Secret
                </button>
            </form>
        </section>
    </main>
    
    <div class="features-list">
        <div class="feature-badge">
            <i class="fas fa-puzzle-piece"></i>
            <span>True Shamir K-of-N</span>
        </div>
        <div class="feature-badge">
            <i class="fas fa-shield-virus"></i>
            <span>Anti-Forensic Noise</span>
        </div>
        <div class="feature-badge">
            <i class="fas fa-clock"></i>
            <span>Time-Locked Access</span>
        </div>
        <div class="feature-badge">
            <i class="fas fa-bomb"></i>
            <span>Real Self-Destruct</span>
        </div>
        <div class="feature-badge">
            <i class="fas fa-eye-slash"></i>
            <span>Zero-Header Mode</span>
        </div>
        <div class="feature-badge">
            <i class="fas fa-fingerprint"></i>
            <span>Carrier Binding</span>
        </div>
        <div class="feature-badge">
            <i class="fas fa-dna"></i>
            <span>Mutation Mode</span>
        </div>
        <div class="feature-badge">
            <i class="fas fa-layer-group"></i>
            <span>Adaptive Bit-Planes</span>
        </div>
    </div>
    
    <!-- Progress Overlay -->
    <div class="progress-overlay" id="progressOverlay">
        <div class="progress-card">
            <div class="progress-spinner"></div>
            <div class="progress-stage" id="progressStage">Processing...</div>
            <div class="progress-detail" id="progressDetail">Please wait</div>
            <div class="progress-bar-container">
                <div class="progress-bar" id="progressBar"></div>
            </div>
            <div class="progress-percent" id="progressPercent">0%</div>
        </div>
    </div>
    
    <!-- Toast -->
    <div class="toast" id="toast">
        <i class="fas fa-check-circle"></i>
        <span id="toastMessage">Operation complete</span>
    </div>
    
    <script>
        // Generate unique task ID
        function generateTaskId() {
            return 'task_' + Math.random().toString(36).substr(2, 16);
        }
        
        // Self-destruct warning toggle
        document.getElementById('f_destruct').addEventListener('change', function() {
            document.getElementById('destructWarning').classList.toggle('active', this.checked);
        });
        
        // Drop zone interactions
        const dropZone = document.getElementById('dropZone');
        ['dragover', 'dragenter'].forEach(event => {
            dropZone.addEventListener(event, e => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            });
        });
        ['dragleave', 'drop'].forEach(event => {
            dropZone.addEventListener(event, e => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
            });
        });
        
        // Poll progress from server
        async function pollProgress(taskId, resolve, reject) {
            const overlay = document.getElementById('progressOverlay');
            const progressBar = document.getElementById('progressBar');
            const progressStage = document.getElementById('progressStage');
            const progressDetail = document.getElementById('progressDetail');
            const progressPercent = document.getElementById('progressPercent');
            
            try {
                const response = await fetch('/progress/' + taskId);
                const data = await response.json();
                
                progressStage.textContent = data.stage || 'Processing...';
                progressDetail.textContent = data.detail || '';
                const percent = data.percent || 0;
                progressBar.style.width = percent + '%';
                progressPercent.textContent = percent + '%';
                
                // Continue polling if not complete
                if (percent < 100 && overlay.classList.contains('active')) {
                    setTimeout(() => pollProgress(taskId, resolve, reject), 300);
                } else if (percent >= 100) {
                    resolve();
                }
            } catch (e) {
                // Continue polling on error (server might be busy)
                if (overlay.classList.contains('active')) {
                    setTimeout(() => pollProgress(taskId, resolve, reject), 500);
                }
            }
        }
        
        // Form submission with progress
        async function submitWithProgress(form, action) {
            const overlay = document.getElementById('progressOverlay');
            const progressBar = document.getElementById('progressBar');
            const progressStage = document.getElementById('progressStage');
            const progressDetail = document.getElementById('progressDetail');
            const progressPercent = document.getElementById('progressPercent');
            
            // Generate and set task ID
            const taskId = generateTaskId();
            const taskIdInput = form.querySelector('input[name="task_id"]');
            if (taskIdInput) {
                taskIdInput.value = taskId;
            }
            
            overlay.classList.add('active');
            progressBar.style.width = '0%';
            progressStage.textContent = 'Uploading files...';
            progressDetail.textContent = '';
            progressPercent.textContent = '0%';
            
            const formData = new FormData(form);
            
            // Start polling for progress
            const progressPromise = new Promise((resolve, reject) => {
                setTimeout(() => pollProgress(taskId, resolve, reject), 200);
            });
            
            try {
                const response = await fetch(action, {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const error = await response.json().catch(() => ({ error: 'Unknown error' }));
                    throw new Error(error.error || 'Operation failed');
                }
                
                progressStage.textContent = 'Downloading result...';
                progressBar.style.width = '98%';
                progressPercent.textContent = '98%';
                
                const blob = await response.blob();
                const filename = response.headers.get('Content-Disposition')?.match(/filename="?([^"]+)"?/)?.[1] 
                    || (action.includes('embed') ? 'stego_shares.zip' : 'extracted_secret');
                
                progressBar.style.width = '100%';
                progressPercent.textContent = '100%';
                progressStage.textContent = 'Complete!';
                
                // Trigger download
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                showToast('Operation completed successfully!', 'success');
                
                setTimeout(() => {
                    overlay.classList.remove('active');
                }, 1500);
                
            } catch (error) {
                overlay.classList.remove('active');
                showToast(error.message, 'error');
            }
        }
        
        // Toast notification
        function showToast(message, type) {
            const toast = document.getElementById('toast');
            const toastMessage = document.getElementById('toastMessage');
            const icon = toast.querySelector('i');
            
            toastMessage.textContent = message;
            toast.className = 'toast ' + type;
            icon.className = type === 'success' ? 'fas fa-check-circle' : 'fas fa-exclamation-circle';
            
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 5000);
        }
        
        // Attach form handlers
        document.getElementById('embedForm').addEventListener('submit', function(e) {
            e.preventDefault();
            submitWithProgress(this, '/embed');
        });
        
        document.getElementById('extractForm').addEventListener('submit', function(e) {
            e.preventDefault();
            submitWithProgress(this, '/extract');
        });
    </script>
</body>
</html>
"""

def main():
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(32)
    app.register_blueprint(veil_bp)
    print("="*60)
    print("  ZYLO VEIL v3 — Advanced Steganography System")
    print("="*60)
    print("  Features:")
    print("    ✓ True Shamir K-of-N (ciphertext splitting)")
    print("    ✓ Real Self-Destruct (irreversible poisoning)")
    print("    ✓ Capacity-Aware Header Embedding")
    print("    ✓ Hardened Stateless Mode (keyed markers)")
    print("    ✓ Stego Mutation Mode")
    print("    ✓ Anti-Forensic Camera Noise Injection")
    print("    ✓ Adaptive Bit-Plane Migration")
    print("="*60)
    print(f"  Server starting on http://0.0.0.0:5000")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    main()
