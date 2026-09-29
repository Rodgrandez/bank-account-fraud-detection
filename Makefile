PY ?= python
.PHONY: data tune evaluate report all test lint
data:     ; $(PY) -m fraud.pipeline data
tune:     ; $(PY) -m fraud.pipeline tune
evaluate: ; $(PY) -m fraud.pipeline evaluate
report:   ; $(PY) -m fraud.pipeline report
all:      ; $(PY) -m fraud.pipeline all
test:     ; $(PY) -m pytest -q
lint:     ; ruff check src tests
