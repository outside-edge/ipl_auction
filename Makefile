.PHONY: all data assembly performance integration verify analysis clean help forecast t20i diagnostics

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
	@echo "  integration Match player names between auction and performance data"
	@echo "  verify      Run data consistency checks"
	@echo "  analysis    Run hedonic regressions and generate predictions"
	@echo "  forecast    Run WAR forecasting pipeline (T20I data + XGBoost model)"
	@echo "  t20i        Download and process T20I data only"
	@echo "  diagnostics Generate unmatched player diagnostics"
	@echo "  clean       Remove generated analysis files"
	@echo ""

# Data collection and assembly
data: assembly performance

assembly: scripts/auction/02_assemble_auction_data.py
	$(PYTHON) scripts/auction/02_assemble_auction_data.py

performance: scripts/perf/01_process_ipl_deliveries.py scripts/perf/02_compute_ipl_war.py
	$(PYTHON) scripts/perf/01_process_ipl_deliveries.py
	$(PYTHON) scripts/perf/02_compute_ipl_war.py

# Data integration
integration: scripts/analysis/01_match_player_names.py
	$(PYTHON) scripts/analysis/01_match_player_names.py

# Verification
verify: scripts/analysis/08_verify_data_consistency.py
	$(PYTHON) scripts/analysis/08_verify_data_consistency.py

# Analysis
analysis: scripts/analysis/05_hedonic_regression.py scripts/analysis/06_identify_duds.py scripts/analysis/07_predict_duds.py
	$(PYTHON) scripts/analysis/05_hedonic_regression.py
	$(PYTHON) scripts/analysis/06_identify_duds.py
	$(PYTHON) scripts/analysis/07_predict_duds.py

# T20I data pipeline
t20i: scripts/perf/03_download_t20i.py scripts/perf/04_process_t20i_deliveries.py scripts/perf/05_compute_t20i_war.py
	$(PYTHON) scripts/perf/03_download_t20i.py
	$(PYTHON) scripts/perf/04_process_t20i_deliveries.py
	$(PYTHON) scripts/perf/05_compute_t20i_war.py

# WAR forecasting pipeline
forecast: t20i scripts/analysis/02_build_player_master.py scripts/analysis/03_build_auction_features.py scripts/analysis/04_train_war_forecast.py
	$(PYTHON) scripts/analysis/02_build_player_master.py
	$(PYTHON) scripts/analysis/03_build_auction_features.py
	$(PYTHON) scripts/analysis/04_train_war_forecast.py

# Diagnostics for unmatched players
diagnostics: scripts/analysis/09_generate_diagnostics.py
	$(PYTHON) scripts/analysis/09_generate_diagnostics.py

# Scrape latest auction (run manually when needed)
scrape-2026:
	$(PYTHON) scripts/auction/01_scrape_auction_2026.py

# Clean generated files
clean:
	rm -f tabs/regression_results.txt
	rm -f tabs/worst_bets.csv
	rm -f tabs/predicted_duds_2026.csv
	rm -f data/analysis/diagnostics/verification_report.md
	rm -f data/analysis/predictions/forecast_evaluation.txt
	rm -f data/analysis/predictions/war_predictions.csv
	rm -f data/analysis/joined/auction_features*.csv
	rm -rf data/perf/t20i/
	rm -rf models/
