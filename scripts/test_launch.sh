CEKEY="0d350ffe89b1e657d439080bf20d526bbcea11b64db94b943b55c156f2e8ddd9"
LP="/global/homes/c/cbu/fw_config/my_launchpad.yaml"

for FE in $(seq 1 255); do
  MG=$((256 - FE))
  pe-extend-wl \
    --launchpad "$LP" \
    --ce-key "$CEKEY" \
    --bin-width 0.2 \
    --composition-counts "Fe:${FE},Mg:${MG}" \
    --step-type swap \
    --check-period 5000 \
    --update-period 1 \
    --seed 0 \
    --samples-per-bin 0 \
    --category cpu \
    --steps-to-run 100000000 \
    --repeats 5 \
    --json
done
