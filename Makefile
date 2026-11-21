IMAGE ?= localhost/rwx-blast-periodic2ru:local
SAMPLE ?= $(abspath ../sample-kit/sample)
SCORER ?= $(abspath ../sample-kit/score.py)
OUTPUT ?= /tmp/rwx-blast-output.jsonl

RUN_FLAGS := --rm --platform linux/amd64 --network none --cpus 2 \
	--memory 4096m --memory-swap 4096m --pids-limit 256 --read-only \
	--tmpfs /tmp:rw,noexec,nosuid,size=64m,mode=1777 --cap-drop ALL \
	--security-opt no-new-privileges --security-opt label=disable \
	--user 65534:65534 --env HOME=/tmp \
	--volume $(SAMPLE)/commits:/data/commits:ro \
	--volume $(SAMPLE)/tests.json:/data/tests.json:ro

.PHONY: build sample check

build:
	docker build --platform linux/amd64 --tag $(IMAGE) .

sample: build
	docker run $(RUN_FLAGS) $(IMAGE) > $(OUTPUT)
	python3 $(SCORER) $(SAMPLE) $(SAMPLE)/answers.json $(OUTPUT)

check: build
	docker run $(RUN_FLAGS) $(IMAGE) > $(OUTPUT)
	python3 $(SCORER) $(SAMPLE) $(SAMPLE)/answers.json $(OUTPUT) > /dev/null && echo "format ok"
