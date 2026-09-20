# python-control-vector

A control-vector artifact exactly as `interp/steering/export.py` writes it: the
safetensors image of one direction per steered block and its JSON manifest, held to
`schemas/control_vector_artifact.json`. Written by `../export_interp_fixtures.py` with
dyadic values so `runtime/tests/python_exporter_contract.rs` compares for equality.
