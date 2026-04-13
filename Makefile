.PHONY: all run data verify retrospective prediction clean help auction perf t20i backtest analysis download-auction download-perf download

PYTHON := python3

all: data verify retrospective prediction

run: all

help:
	@echo "IPL Auction Analysis Pipeline"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all            Run full pipeline (data + verify + retrospective + prediction)"
	@echo "  download       Download auction data from Kaggle"
	@echo "  download-auction  Download auction data from Kaggle"
	@echo "  data           Data collection and assembly (auction + perf)"
	@echo "  auction        Assemble auction data"
	@echo "  perf           Process IPL ball-by-ball data into WAR"
	@echo "  t20i           Download and process T20I data"
	@echo "  verify         Join data sources and verify consistency"
	@echo "  retrospective  Run retrospective analysis (hedonic regressions, identify duds)"
	@echo "  prediction     Forward-looking prediction (WAR forecast, predict duds)"
	@echo "  backtest       Run historical backtest of predictions"
	@echo "  analysis       Run comprehensive backtest + economic analysis"
	@echo "  clean          Remove generated analysis files"
	@echo ""

# Download data from Kaggle
download: download-auction

download-auction:
	bash scripts/auction/00_download_kaggle.sh

# Data collection and assembly
data: auction perf

auction:
	$(PYTHON) scripts/auction/02_assemble_auction_data.py

perf:
	@if [ -f data/perf/ipl/player_season_stats.csv ] && [ -f data/perf/ipl/player_season_war.csv ]; then \
		echo "Using cached perf data (data/perf/ipl/*.csv exist)"; \
	else \
		$(PYTHON) scripts/perf/01_process_ipl_deliveries.py; \
		$(PYTHON) scripts/perf/02_compute_ipl_war.py; \
	fi

# T20I data pipeline (optional, for prediction features)
t20i:
	$(PYTHON) scripts/perf/03_download_t20i.py
	$(PYTHON) scripts/perf/04_process_t20i_deliveries.py
	$(PYTHON) scripts/perf/05_compute_t20i_war.py

# Verification & joining (runs after data, creates joined datasets)
verify:
	$(PYTHON) scripts/verify/01_verify_data_consistency.py
	$(PYTHON) scripts/verify/02_match_player_names.py
	$(PYTHON) scripts/verify/03_build_player_master.py
	$(PYTHON) scripts/verify/04_generate_diagnostics.py

# Retrospective analysis
retrospective:
	$(PYTHON) scripts/retrospective/01_hedonic_regression.py
	$(PYTHON) scripts/retrospective/02_identify_duds.py

# Forward-looking prediction
prediction: t20i
	$(PYTHON) scripts/prediction/01_build_auction_features.py
	$(PYTHON) scripts/prediction/02_train_war_forecast.py
	$(PYTHON) scripts/prediction/03_validate_model.py
	$(PYTHON) scripts/prediction/04_predict_duds.py

# Backtest (optional, manual run)
backtest:
	$(PYTHON) scripts/prediction/05_backtest_predictions.py

# Comprehensive backtest + economic analysis
analysis:
	$(PYTHON) scripts/prediction/06_comprehensive_backtest.py
	$(PYTHON) scripts/prediction/07_economic_analysis.py

# Scrape latest auction (run manually when needed)
scrape-2026:
	$(PYTHON) scripts/auction/01_scrape_auction_2026.py

# Clean generated files
clean:
	rm -f tabs/regression_results.txt
	rm -f tabs/worst_bets.csv
	rm -f tabs/predicted_duds_2026.csv
	rm -f tabs/forecast_evaluation.txt
	rm -f tabs/war_predictions.csv
	rm -f tabs/validation_report.txt
	rm -f tabs/backtest_results.csv
	rm -f data/analysis/diagnostics/verification_report.md
	rm -rf data/model/
	rm -rf data/perf/t20i/
	rm -rf models/
