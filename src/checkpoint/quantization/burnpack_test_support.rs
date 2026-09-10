//! Fault-injection helpers compiled only for the library's unit tests.
//!
//! The scoped visibility matches the original parent-module methods: callers
//! remain restricted to checkpoint quantization tests. Moving this inherent
//! implementation does not add fields or change the non-test session API.

use super::*;

impl QuantizedBurnpackSession {
    /// Returns the private staging path for isolated filesystem fault fixtures.
    pub(in crate::checkpoint::quantization) fn staging_path_for_test(&self) -> &Path {
        &self.staging_path
    }

    /// Reads the staging file length before a test deliberately corrupts it.
    pub(in crate::checkpoint::quantization) fn staging_len_for_test(&self) -> u64 {
        self.staging_file
            .as_ref()
            .unwrap()
            .metadata()
            .unwrap()
            .len()
    }

    /// Changes only a fixture's staging length to exercise publication guards.
    pub(in crate::checkpoint::quantization) fn truncate_staging_for_test(&mut self, length: u64) {
        self.staging_file.as_ref().unwrap().set_len(length).unwrap();
    }

    /// Flips selected bits in one staged byte to exercise publication verification.
    pub(in crate::checkpoint::quantization) fn xor_staging_byte_for_test(
        &mut self,
        offset: u64,
        mask: u8,
    ) {
        let file = self.staging_file.as_mut().unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        let mut byte = [0_u8; 1];
        file.read_exact(&mut byte).unwrap();
        byte[0] ^= mask;
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(&byte).unwrap();
        file.seek(SeekFrom::End(0)).unwrap();
    }
}
