//! Executable checks of the Verus-verified feed-forward width validation.

use super::*;

/// Both widths must be non-zero, and the residual width is reported first.
#[test]
fn width_faults_report_the_first_empty_width() {
    assert_eq!(width_fault(8, 32), None);
    assert_eq!(width_fault(0, 32), Some(WidthFault::HiddenSize));
    assert_eq!(width_fault(0, 0), Some(WidthFault::HiddenSize));
    assert_eq!(width_fault(8, 0), Some(WidthFault::IntermediateSize));
    assert_eq!(width_fault(usize::MAX, usize::MAX), None);
}
