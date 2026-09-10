# Historical public-example dependency lock

`Cargo.lock` and `examples/` preserve the exact bytes whose SHA-256 is recorded in
[the September 9 public-example measurement](../public-examples-20260909.json).
The archived lock was recovered from `cc56333b4bd35083013f85406defb483d9a737aa:Cargo.lock` during v0.3 work;
its SHA-256 is `819b26befc5ff5040f5ca42b5088826652bd0e713935e56bf8ab30e0b1cc381d` and matches the existing measurement.
The original pre-extraction commit is named in that record but is not available
as a Git object in this standalone checkout. The measurement itself is unchanged.

The two example files were retained before documentation changes, after their
SHA-256 values were verified against the unchanged measurement. Current code
and dependencies may change independently. The historical record test
authenticates these retained inputs, rather than requiring every future
checkout to use the September 9 source and dependencies. This is archival provenance,
not a rerun or qualification of the new training dependency graph.
