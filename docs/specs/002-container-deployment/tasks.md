# Feature 002 — Tasks

- [x] **T001** `docker/Dockerfile`:
  - NGC base image; copy the app packages
  - `PIP_CONSTRAINT="" pip install` the deps, without torch
  - `mkdir /outputs`; `EXPOSE 8500`
  - socket `HEALTHCHECK`; `ENV` defaults; `CMD`

  → FR-002-01, 02, 05, 06, 09
- [x] **T002** `docker/docker-compose.yml`:
  - build context `..`
  - port 8500; HF cache bind mount, `HF_HOME`, the `image-outputs` volume
  - env passthrough
  - GPU reservation, `ipc: host`, `shm_size`, `memlock`
  - socket healthcheck; `restart: no`

  *(depends T001)* → FR-002-03–07, 09
- [x] **T003** `docker/docker-compose.override.yml`: external `ai-bridge` network. *(depends T002)* → FR-002-08
- [x] **T004** README Docker quick start (both compose invocations; verifying with `docker ps` and `curl`).
- [x] **T005** Manual acceptance run of AS-1 to AS-6 on the DGX Spark, with the results recorded in the CHANGELOG. *(Done 2026-10-18 in the 006 GPU window. All pass after fix F-3; see the traceability table.)*
