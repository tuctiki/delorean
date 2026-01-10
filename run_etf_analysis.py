import qlib
from constants import QLIB_PROVIDER_URI, QLIB_REGION
from etf_strategy import ETFStrategy

# Initialize Qlib
qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)

if __name__ == "__main__":
    strategy = ETFStrategy()
    strategy.run()
