import { useState, useEffect, useRef } from 'react';
import useSWR from 'swr';
import { Play, Activity, TrendingUp, AlertTriangle, RefreshCw } from 'lucide-react';
import styles from '../styles/Home.module.css';

const fetcher = (...args) => fetch(...args).then(res => res.json());

export default function Home() {
  const { data: recs, error: recError } = useSWR('http://localhost:8000/api/recommendations', fetcher, { refreshInterval: 5000 });
  const { data: status, error: statusError } = useSWR('http://localhost:8000/api/status', fetcher, { refreshInterval: 1000 });

  const [isRunning, setIsRunning] = useState(false);
  const consoleRef = useRef(null);

  // Auto scroll console
  useEffect(() => {
    if (consoleRef.current) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [status?.log]);

  const handleRun = async () => {
    setIsRunning(true);
    try {
      await fetch('http://localhost:8000/api/run-daily', { method: 'POST' });
    } catch (e) {
      console.error(e);
      setIsRunning(false);
    }
  };

  // Sync local running state with backend status
  useEffect(() => {
    if (status) {
      setIsRunning(status.running);
    }
  }, [status]);

  const isMarketBull = recs?.market_status === 'Bull';

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Dashboard</h1>

      <div className={styles.grid}>
        {/* Daily Task Card */}
        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <h2><Activity size={20} /> Daily Task</h2>
            <button
              className={styles.runButton}
              onClick={handleRun}
              disabled={isRunning}
            >
              {isRunning ? 'Running...' : 'Run Daily Update'} <Play size={16} />
            </button>
          </div>
          <div className={styles.console} ref={consoleRef}>
            {status?.log || "Ready to run..."}
          </div>
        </div>

        {/* Strategy Config Card */}
        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <h2><RefreshCw size={20} /> Active Strategy</h2>
          </div>
          <div className={styles.stats}>
            <div className={styles.statRow}>
              <span>Mode</span>
              <strong>{recs?.strategy_config?.mode || 'Standard'}</strong>
            </div>
            <div className={styles.statRow}>
              <span>Top K / Buffer</span>
              <strong>{recs?.strategy_config?.topk || 4} / {recs?.strategy_config?.buffer || 0}</strong>
            </div>
            <div className={styles.statRow}>
              <span>Smoothing</span>
              <strong>{recs?.strategy_config?.smooth_window || 10}d</strong>
            </div>
          </div>
        </div>

        {/* Market Status Card */}
        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <h2><TrendingUp size={20} /> Market Status</h2>
            <div className={`${styles.badge} ${isMarketBull ? styles.bull : styles.bear}`}>
              {recs?.market_status || 'Unknown'}
            </div>
          </div>
          <div className={styles.stats}>
            <div className={styles.statRow}>
              <span>Benchmark Close</span>
              <strong>{recs?.market_data?.benchmark_close?.toFixed(2) || '-'}</strong>
            </div>
            <div className={styles.statRow}>
              <span>Benchmark MA60</span>
              <strong>{recs?.market_data?.benchmark_ma60?.toFixed(2) || '-'}</strong>
            </div>
          </div>
          {!isMarketBull && recs && (
            <div className={styles.alert}>
              <AlertTriangle size={16} /> Bear Market. Cash Recommended.
            </div>
          )}
        </div>

        {/* Recommendations Card */}
        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <h2>Top Recommendations</h2>
            <span className={styles.date}>{recs?.date || ''}</span>
          </div>
          <div className={styles.list}>
            {/* Header Row */}
            {!recError && recs?.top_recommendations?.length > 0 && (
              <div className={styles.listHeader}>
                <div className={styles.rank} title="Rank based on Model Score">#</div>
                <div className={styles.symbol} title="ETF Ticker Symbol">ETF</div>
                <div className={styles.allocation} title="Recommended Weight (Risk Parity)">Alloc</div>
                <div className={styles.score} title="Raw Model Prediction Score">Score</div>
                <div style={{ marginLeft: '1rem', width: '60px', fontSize: '0.85rem', color: '#a0a0a0' }} title="20-Day Annualized Volatility">Vol</div>
              </div>
            )}

            {recError ? <div className={styles.placeholder}>Error loading data</div> : null}
            {!recError && recs?.top_recommendations?.length > 0 ? (
              recs.top_recommendations.map((item, i) => (
                <div key={item.symbol} className={`${styles.listItem} ${item.is_buffer ? styles.bufferItem : ''}`}>
                  <div className={`${styles.rank} ${item.is_buffer ? styles.rankBuffer : ''}`}>
                    #{item.rank || i + 1}
                  </div>
                  <div className={styles.symbol}>{item.symbol}</div>

                  {/* Allocation Column */}
                  <div className={styles.allocation}>
                    {item.target_weight ? `${(item.target_weight * 100).toFixed(1)}%` : '-'}
                  </div>

                  <div className={styles.score}>{item?.score?.toFixed(4)}</div>
                  {item.volatility !== undefined && (
                    <div className={styles.volatility} title="20-Day Volatility">
                      v{(item.volatility * 100).toFixed(2)}%
                    </div>
                  )}
                </div>
              ))
            ) : (
              <div className={styles.placeholder}>No recommendations. Run daily task.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
