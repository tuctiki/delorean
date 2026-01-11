import qlib
from qlib.config import REG_CN
from constants import QLIB_PROVIDER_URI, QLIB_REGION
from qlib.contrib.data.handler import Alpha158

try:
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    a158 = Alpha158(instruments=['510300.SH'], start_time='2020-01-01', end_time='2020-01-10')
    conf = a158.get_feature_config()
    print("Alpha158 Feature Config Type:", type(conf))
    if isinstance(conf, (tuple, list)):
        print("Alpha158 Feature Config Length:", len(conf))
        print("Alpha158 Feature Config Item 0 Type:", type(conf[0]))
        print("Alpha158 Feature Config Item 1 Type:", type(conf[1]))
except Exception as e:
    print("Error:", e)
