.PHONY: all data assembly performance integration verify analysis clean help

PYTHON := python3

all: data integration verify analysis

help:
	@echo "IPL Auction Analysis Pipeline"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all         Run full pipeline (data + integration + verify + analysis)"
	@echo "  data        Data collection and assembly"
	@echo "  assembly    Assemble auction data and create player registry"
	@echo "  performance Process ball-by-ball data into season stats and WAR"
	@echo "  integration Match player names and adjust for inflation"
	@echo "  verify      Run data consistency checks"
	@echo "  analysis    Run hedonic regressions and generate predictions"
	@echo "  clean       Remove generated analysis files"
	@echo ""

# Data collection and assembly
data: assembly performance

assembly: scripts/assemble_auction_data.py
	$(PYTHON) scripts/assemble_auction_data.py

performance: scripts/process_deliveries.py scripts/compute_war.py
	$(PYTHON) scripts/process_deliveries.py
	$(PYTHON) scripts/compute_war.py

# Data integration
integration: scripts/match_player_names.py scripts/adjust_inflation.py
	$(PYTHON) scripts/match_player_names.py
	$(PYTHON) scripts/adjust_inflation.py

# Verification
verify: scripts/verify_data_consistency.py
	$(PYTHON) scripts/verify_data_consistency.py

# Analysis
analysis: scripts/hedonic_regression.py scripts/identify_duds.py scripts/predict_duds.py
	$(PYTHON) scripts/hedonic_regression.py
	$(PYTHON) scripts/identify_duds.py
	$(PYTHON) scripts/predict_duds.py

# Scrape latest auction (run manually when needed)
scrape-2026:
	$(PYTHON) scripts/scrape_auction_2026.py

# Clean generated files
clean:
	rm -f data/analysis/regression_results.txt
	rm -f data/analysis/worst_bets.csv
	rm -f data/analysis/predicted_duds_2026.csv
	rm -f data/analysis/verification_report.md
	rm -f data/analysis/fig_*.png
