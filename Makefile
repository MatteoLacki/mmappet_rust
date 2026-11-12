UV ?= uv
PYTHON_VERSION = 3.12
# The Python mmappet the oracle tests compare against.
MMAPPET_PY = git+https://github.com/michalsta/mmappet.git@8c6e9a3131e1d1e8c2a0982efa801d84f8191bab

.PHONY: venv test clean-venv

venv: .venv/.stamp

.venv/.stamp: Makefile
	$(UV) venv --allow-existing --python $(PYTHON_VERSION) .venv
	$(UV) pip install --python .venv/bin/python "mmappet @ $(MMAPPET_PY)"
	touch $@

test: venv
	cargo test --all-features

clean-venv:
	rm -rf .venv
