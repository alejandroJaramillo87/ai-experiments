/*
 * npz — minimal writer for numpy .npy arrays inside a .npz (zip) container.
 *
 * Purpose-built for the interp activation-capture shard format
 * (interp/probes/capture.py): float16 activation matrices and int64 label
 * vectors, loadable with np.load. Entries are stored uncompressed — np.load
 * reads stored zip members, and determinism matters more than shard size.
 * Zip timestamps are fixed at zero so identical data produces identical
 * bytes.
 */

use std::io::Write;

use anyhow::Result;

/// npy v1.0 image for a 2-D float16 array in C order.
pub fn npy_f16(rows: usize, cols: usize, data: &[u16]) -> Vec<u8> {
    assert_eq!(data.len(), rows * cols);
    let header = format!("{{'descr': '<f2', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    let mut out = npy_prelude(&header);
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// npy v1.0 image for a 1-D int64 array.
pub fn npy_i64(data: &[i64]) -> Vec<u8> {
    let header = format!(
        "{{'descr': '<i8', 'fortran_order': False, 'shape': ({},), }}",
        data.len()
    );
    let mut out = npy_prelude(&header);
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Magic, version 1.0, and the space-padded header dict. The full prelude is
/// padded to a multiple of 64 bytes and ends in a newline, per the npy
/// format specification.
fn npy_prelude(header: &str) -> Vec<u8> {
    let mut h = header.to_string();
    let unpadded = 10 + h.len() + 1; // magic(6) + version(2) + len(2) + '\n'
    h.push_str(&" ".repeat(unpadded.next_multiple_of(64) - unpadded));
    h.push('\n');
    let mut out = Vec::with_capacity(10 + h.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&u16::try_from(h.len()).expect("short header").to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    out
}

/// IEEE CRC-32 (the zip checksum), bitwise; shard writing is not a hot path.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Accumulates named npy images and writes them as one stored-entry zip.
#[derive(Default)]
pub struct NpzWriter {
    entries: Vec<(String, Vec<u8>)>,
}

impl NpzWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an array under `key` (np.load exposes it under this name; the
    /// `.npy` suffix is appended here).
    pub fn add(&mut self, key: &str, npy: Vec<u8>) {
        self.entries.push((format!("{key}.npy"), npy));
    }

    pub fn write_to(self, path: &std::path::Path) -> Result<()> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(&self.finish())?;
        Ok(())
    }

    pub fn finish(self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut central = Vec::new();
        for (name, data) in &self.entries {
            let offset = u32::try_from(out.len()).expect("shard under 4GB");
            let crc = crc32(data);
            let size = u32::try_from(data.len()).expect("entry under 4GB");
            let name_len = u16::try_from(name.len()).expect("short name");
            // Local file header: stored, no flags, zero timestamps.
            out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
            out.extend_from_slice(&20u16.to_le_bytes()); // version needed
            out.extend_from_slice(&[0; 8]); // flags, method, time, date
            out.extend_from_slice(&crc.to_le_bytes());
            out.extend_from_slice(&size.to_le_bytes());
            out.extend_from_slice(&size.to_le_bytes());
            out.extend_from_slice(&name_len.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes()); // extra len
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(data);
            // Matching central-directory record.
            central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            central.extend_from_slice(&20u16.to_le_bytes()); // version made by
            central.extend_from_slice(&20u16.to_le_bytes()); // version needed
            central.extend_from_slice(&[0; 8]); // flags, method, time, date
            central.extend_from_slice(&crc.to_le_bytes());
            central.extend_from_slice(&size.to_le_bytes());
            central.extend_from_slice(&size.to_le_bytes());
            central.extend_from_slice(&name_len.to_le_bytes());
            central.extend_from_slice(&[0; 8]); // extra, comment, disk, internal attrs
            central.extend_from_slice(&[0; 4]); // external attrs
            central.extend_from_slice(&offset.to_le_bytes());
            central.extend_from_slice(name.as_bytes());
        }
        let cd_offset = u32::try_from(out.len()).expect("shard under 4GB");
        let cd_size = u32::try_from(central.len()).expect("directory under 4GB");
        let count = u16::try_from(self.entries.len()).expect("few entries");
        out.extend_from_slice(&central);
        // End of central directory.
        out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        out.extend_from_slice(&[0; 4]); // disk numbers
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&cd_size.to_le_bytes());
        out.extend_from_slice(&cd_offset.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // comment len
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_reference_vector() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn npy_prelude_is_64_byte_aligned_and_newline_terminated() {
        let img = npy_i64(&[1, 2, 3]);
        assert_eq!(&img[..6], b"\x93NUMPY");
        let header_len = u16::from_le_bytes([img[8], img[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0);
        assert_eq!(img[10 + header_len - 1], b'\n');
        assert_eq!(img.len(), 10 + header_len + 3 * 8);
    }

    #[test]
    fn python_numpy_loads_a_written_npz() {
        // Held to the end of the test: the directory is removed when the guard
        // drops, so a failed assertion below cannot leak the shard.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shard.npz");
        let mut w = NpzWriter::new();
        // 2x3 f16 matrix [[0, 0.5, 1], [1.5, -2, 4]] and labels [7, -1].
        let f16s: Vec<u16> = [0.0f32, 0.5, 1.0, 1.5, -2.0, 4.0]
            .iter()
            .map(|&v| crate::tensor::f32_to_f16(v))
            .collect();
        w.add("layer_3", npy_f16(2, 3, &f16s));
        w.add("labels", npy_i64(&[7, -1]));
        w.write_to(&path).unwrap();
        let check = format!(
            "import numpy as np\n\
             z = np.load('{}')\n\
             a = z['layer_3']\n\
             assert a.dtype == np.float16 and a.shape == (2, 3), (a.dtype, a.shape)\n\
             assert a.tolist() == [[0, 0.5, 1], [1.5, -2, 4]], a.tolist()\n\
             l = z['labels']\n\
             assert l.dtype == np.int64 and l.tolist() == [7, -1], l\n",
            path.display()
        );
        let out = std::process::Command::new("python3")
            .args(["-c", &check])
            .output()
            .expect("python3 available in the sandbox");
        assert!(
            out.status.success(),
            "np.load rejected the shard: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
