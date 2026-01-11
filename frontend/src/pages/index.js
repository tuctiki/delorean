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
            {recError ? <div className={styles.placeholder}>Error loading data</div> : null}
            {!recError && recs?.top_recommendations?.length > 0 ? (
              recs.top_recommendations.map((item, i) => (
                <div key={item.symbol} className={styles.listItem}>
                  <div className={styles.rank}>#{i + 1}</div>
                  <div className={styles.symbol}>{item.symbol}</div>
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
