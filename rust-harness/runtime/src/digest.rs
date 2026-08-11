/*
 * digest — SHA-256, for artifacts that have to prove they were not edited.
 *
 * Hand-rolled rather than pulled in, because the workspace's dependency
 * posture admits format crates and little else (.styles/RUST_STYLE.md), and a
 * hash whose test vectors are published is the one kind of algorithm a
 * transcription can be proved right about. The FIPS 180-4 vectors are
 * asserted below.
 *
 * It lives here rather than in the binary that first needed it because two
 * callers now hash artifacts for the same reason -- activation-capture
 * records what it read, and the ladder's fixture manifest records what was
 * generated -- and two copies of a hash is two chances to have one of them
 * wrong.
 */

/// Lowercase hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finish()
}

/// Lowercase hex SHA-256 of a file, read in fixed-size chunks.
///
/// For the callers whose input is a multi-gigabyte checkpoint they no longer
/// hold. `CpuModel::load` drops the weights mapping once the model is built,
/// so an identity check running afterwards has to go back to the file -- and
/// reading it whole to hash it would reintroduce, transiently, the residency
/// the drop exists to remove. Peak memory here is `CHUNK` whatever the file's
/// size.
pub fn sha256_file(path: &std::path::Path) -> std::io::Result<String> {
    use std::io::Read;
    const CHUNK: usize = 1 << 20;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            return Ok(hasher.finish());
        }
        hasher.update(&buf[..n]);
    }
}

/// FIPS 180-4 SHA-256 state, fed incrementally.
///
/// Incremental rather than one-shot so `sha256_file` can exist. The one-shot
/// form this replaced began with `bytes.to_vec()` in order to append its
/// padding, so hashing the E4B weights allocated a second copy of all 15 GB --
/// a cost nobody had looked for, because the visible callers passed a dataset
/// and a short string.
struct Sha256 {
    h: [u32; 8],
    /// Bytes of the current block not yet compressed, always under 64.
    block: [u8; 64],
    filled: usize,
    /// Total bytes consumed, for the length suffix the padding carries.
    len: u64,
}

impl Sha256 {
    fn new() -> Sha256 {
        Sha256 {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            block: [0; 64],
            filled: 0,
            len: 0,
        }
    }

    fn update(&mut self, mut bytes: &[u8]) {
        self.len = self.len.wrapping_add(bytes.len() as u64);
        while !bytes.is_empty() {
            let take = (64 - self.filled).min(bytes.len());
            self.block[self.filled..self.filled + take].copy_from_slice(&bytes[..take]);
            self.filled += take;
            bytes = &bytes[take..];
            if self.filled == 64 {
                let block = self.block;
                compress(&mut self.h, &block);
                self.filled = 0;
            }
        }
    }

    fn finish(mut self) -> String {
        // FIPS 180-4 padding: a 1 bit, zeroes, then the length in bits as a
        // big-endian u64. Emitted through `update` so the buffering is not
        // written twice, which is why `len` is captured before it moves.
        let bit_len = self.len.wrapping_mul(8);
        self.update(&[0x80]);
        while self.filled != 56 {
            self.update(&[0]);
        }
        self.update(&bit_len.to_be_bytes());
        debug_assert_eq!(self.filled, 0, "padding left a partial block");
        self.h.iter().map(|v| format!("{v:08x}")).collect()
    }
}

/// One SHA-256 block, folded into `h`.
fn compress(h: &mut [u32; 8], block: &[u8; 64]) {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut w = [0u32; 64];
    for (i, c) in block.chunks_exact(4).enumerate() {
        w[i] = u32::from_be_bytes(c.try_into().unwrap());
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let t1 = hh
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    for (s, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
        *s = s.wrapping_add(v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_the_fips_test_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    /// Sizes chosen around the 64-byte block and the 1 MiB read buffer, which
    /// is where an incremental hash gets it wrong: a block boundary landing
    /// mid-`update`, the 55/56/64 padding cases where the length suffix does
    /// not fit in the final block, and a payload spanning several reads.
    #[test]
    fn hashing_a_file_agrees_with_hashing_its_bytes_at_every_awkward_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payload");
        for len in [
            0,
            1,
            55,
            56,
            63,
            64,
            65,
            119,
            120,
            1 << 20,
            (1 << 20) + 7,
            3_000_000,
        ] {
            // A varying pattern rather than a constant: a digest over zeroes
            // would agree with a truncated read for the wrong reason.
            let bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            std::fs::write(&path, &bytes).unwrap();
            assert_eq!(
                sha256_file(&path).unwrap(),
                sha256_hex(&bytes),
                "streamed and one-shot digests disagree over {len} bytes"
            );
        }
    }

    /// `update` called with a partial block already buffered.
    ///
    /// `sha256_file` normally hands over whole 1 MiB chunks, so this path only
    /// runs when the kernel returns a short read -- rare, unreproducible on
    /// demand, and silently wrong if the carry-over arithmetic is off. Driving
    /// `Sha256` directly is the only way to pin it.
    #[test]
    fn hashing_in_irregular_chunks_agrees_with_one_shot() {
        let bytes: Vec<u8> = (0..700).map(|i| (i % 251) as u8).collect();
        // Deliberately coprime with 64 and varying, so a block boundary falls
        // inside a chunk rather than between two.
        for step in [1, 3, 7, 13, 31, 63, 64, 65, 100] {
            let mut hasher = Sha256::new();
            for chunk in bytes.chunks(step) {
                hasher.update(chunk);
            }
            assert_eq!(
                hasher.finish(),
                sha256_hex(&bytes),
                "chunked digest disagrees at step {step}"
            );
        }
    }

    #[test]
    fn hashing_a_missing_file_names_it() {
        let err = sha256_file(std::path::Path::new("/nonexistent/model.safetensors"))
            .expect_err("a checkpoint that vanished after load is not a silent skip");
        assert_eq!(
            err.kind(),
            std::io::ErrorKind::NotFound,
            "a caller distinguishes a missing checkpoint from an unreadable one"
        );
    }

    #[test]
    fn sha256_spans_the_padding_boundary() {
        // 55, 56 and 64 bytes: the last message that fits one block with its
        // length, the first that needs a second, and an exact block. Padding
        // is where a hand-rolled implementation goes wrong, and the two
        // vectors above are both short enough to miss it entirely.
        assert_eq!(
            sha256_hex(&[b'a'; 55]),
            "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"
        );
        assert_eq!(
            sha256_hex(&[b'a'; 56]),
            "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"
        );
        assert_eq!(
            sha256_hex(&[b'a'; 64]),
            "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"
        );
    }
}
