#!/usr/bin/env bash
# 把 TEAM_PLAN.md 第 9 节列出的论文下载到 papers/（在 cfm_project 根目录下运行：bash scripts/download_papers.sh）
set -u
cd "$(dirname "$0")/../" && mkdir -p papers && cd papers
while read -r id name; do
  [ -z "$id" ] && continue
  [ -s "$name.pdf" ] && { echo "skip $name"; continue; }
  echo "get  $name"
  curl -sL --retry 2 -A "Mozilla/5.0" -o "$name.pdf" "https://arxiv.org/pdf/$id"
  sleep 1
done <<'EOF'
2208.11970 00_Luo2022_Understanding_Diffusion_Unified
2412.06264 01_Lipman2024_Flow_Matching_Guide_and_Code
2006.11239 10_Ho2020_DDPM
2010.02502 11_Song2021_DDIM
1907.05600 12_Song2019_NCSN_Score_Matching
2011.13456 13_Song2021_Score_SDE
2102.09672 14_Nichol2021_Improved_DDPM
2207.12598 15_Ho2022_Classifier_Free_Guidance
2206.00364 16_Karras2022_EDM
2210.02747 20_Lipman2023_Flow_Matching
2209.03003 21_Liu2023_Rectified_Flow
2209.15571 22_Albergo2023_Stochastic_Interpolants
2302.00482 23_Tong2024_OT_CFM_Minibatch_OT
2307.03672 24_Tong2024_SF2M_Schrodinger_Bridge
1806.07366 25_Chen2018_Neural_ODE
2403.03206 30_Esser2024_SD3_Rectified_Flow_Transformers
2303.01469 31_Song2023_Consistency_Models
2505.13447 32_Geng2025_MeanFlow
1706.08500 40_Heusel2017_FID
1312.6114 41_Kingma2014_VAE
EOF
echo "done: $(ls *.pdf | wc -l) PDFs in papers/"
