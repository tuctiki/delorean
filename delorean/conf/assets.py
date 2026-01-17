
# ETF List
ETF_LIST = [
# Broad Market (China)
    "510300.SH",    # 沪深300ETF
    "563360.SH",    # 华泰柏瑞中证A500ETF
    "159915.SZ",    # 新增：创业板ETF (ChiNext)
    "588000.SH",    # 新增：科创50ETF (STAR 50)
    "512100.SH",    # 新增：中证1000ETF (Small Cap)
    "510050.SH",    # 新增：上证50 (Large Cap Value)
    "510900.SH",    # 新增：H股ETF (H-Share)
    
    # Global / QDII (Available but segregated from main model)
    # "513050.SH",    # KWEB
    # "513100.SH",    # Nasdaq 100
    # "513500.SH",    # S&P 500
    # "159920.SZ",    # HSI
    
    # Commodities
    "518880.SH",    # 新增：黄金ETF (Gold)
    "512400.SH",    # 新增：有色金属 (Commodity Proxy)
    "515220.SH",    # 新增：煤炭ETF (Coal)
    "159881.SZ",    # 新增：有色60ETF (Non-Ferrous 60)
    "516780.SH",    # 新增：稀土ETF (Rare Earth)
    "515210.SH",    # 新增：钢铁ETF (Steel)
    "159930.SZ",    # 新增：能源ETF (Energy)
    "516020.SH",    # 新增：化工ETF (Chemical)
    
    # Sector - Healthcare
    "512010.SH",    # 医药
    "512170.SH",    # 新增：医疗ETF (Medical Device/Service)
    "159992.SZ",    # 新增：创新药ETF (Innovative Pharma)
    "560080.SH",    # 新增：中药ETF (TCM)
    "512290.SH",    # 新增：生物医药ETF (Bio-Pharma)

    # Sector - Consumer
    "512690.SH",    # 白酒
    "515170.SH",    # 新增：食品饮料ETF (Food & Beverage)
    "159996.SZ",    # 新增：家电ETF (Home Appliances)
    "159865.SZ",    # 新增：养殖ETF (Breeding)
    "159867.SZ",    # 新增：畜牧ETF (Livestock)

    # Sector - Tech & Others
    "512480.SH",    # 半导体
    "516160.SH",    # 新能源车
    "512800.SH",    # 银行
    "510630.SH",    # 消费 (Legacy)
    "515790.SH",    # 光伏
    "512880.SH",    # 证券
    "512660.SH",    # 新增：军工 (Defense)
    "510880.SH",    # 红利ETF (Defensive)
]

ETF_NAME_MAP = {
    # Broad Market (China)
    "510300.SH": "CSI 300",
    "563360.SH": "A500",
    "159915.SZ": "ChiNext (Startup)",
    "588000.SH": "STAR 50 (Tech)",
    "512100.SH": "CSI 1000 (Small Cap)",
    "510050.SH": "SSE 50 (Large Cap)",
    "510900.SH": "H-Share (HSCEI)",
    
    # Global / QDII
    "513050.SH": "KWEB (China Internet)",
    "513100.SH": "Nasdaq 100",
    "513500.SH": "S&P 500",
    "159920.SZ": "Hang Seng Index",
    
    # Commodities
    "518880.SH": "Gold",
    "512400.SH": "Non-Ferrous (Resources)",
    "515220.SH": "Coal",
    "159881.SZ": "Non-Ferrous 60",
    "516780.SH": "Rare Earth",
    "515210.SH": "Steel",
    "159930.SZ": "Energy",
    "516020.SH": "Chemical",
    
    # Healthcare
    "512010.SH": "Pharma",
    "512170.SH": "Medical Device",
    "159992.SZ": "Innovative Pharma",
    "560080.SH": "TCM (Chinese Med)",
    "512290.SH": "Bio-Pharma",

    # Consumer
    "512690.SH": "Spirit/Liquor",
    "515170.SH": "Food & Beverage",
    "159996.SZ": "Home Appliances",
    "159865.SZ": "Breeding (Pork)",
    "159867.SZ": "Livestock",
    "510630.SH": "Consumer (Legacy)",

    # Tech & Others
    "512480.SH": "Semiconductor",
    "516160.SH": "New Energy",
    "512800.SH": "Bank",
    "515790.SH": "PV/Solar",
    "512880.SH": "Securities",
    "512660.SH": "Defense/Military",
    "510880.SH": "Dividend (RedChip)",
}

# Benchmark
BENCHMARK = "510300.SH"
