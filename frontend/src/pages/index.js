import { useState, useEffect, useRef } from 'react';
import useSWR from 'swr';
import { Play, Activity, TrendingUp, AlertTriangle, RefreshCw } from 'lucide-react';
import Layout from '../components/Layout';
import AllocationChart from '../components/AllocationChart';
import styles from '../styles/Home.module.css';

const fetcher = (...args) => fetch(...args).then(res => res.json());

export default function Home() {
  const { data: recs, error: recError } = useSWR('http://localhost:8000/api/recommendations', fetcher, { refreshInterval: 5000 });
  const { data: status, error: statusError } = useSWR('http://localhost:8000/api/status', fetcher, { refreshInterval: 1000 });

  const [isRunning, setIsRunning] = useState(false);
  const [capital, setCapital] = useState(100000);
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


    <>
      <div className={styles.grid}>
        <div style={{ gridColumn: '1 / -1', marginBottom: '10px' }}>
          <h1 className={styles.title}>Dashboard</h1>
        </div>


        {/* Row 1: Status & Controls */}
        <div className={`${styles.card} ${styles.col4}`}>
          <div className={styles.cardHeader}>
            <h2><Activity size={20} /> Daily Task</h2>
            <button
              className={styles.runButton}
              onClick={handleRun}
              disabled={isRunning}
            >
              {isRunning ? 'Running...' : 'Run'} <Play size={16} />
            </button>
          </div>
          <div className={styles.console} ref={consoleRef}>
            {status?.log || "Ready to run..."}
          </div>
        </div>

        <div className={`${styles.card} ${styles.col4}`}>
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
            <div className={styles.statRow}>
              <span>Label Horizon</span>
              <strong>{recs?.strategy_config?.label_horizon || 1}d</strong>
            </div>
          </div>
        </div>

        <div className={`${styles.card} ${styles.col4}`}>
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

        {/* Row 2: Recommendations List (Left) */}
        <div className={`${styles.card} ${styles.col8}`}>
          <div className={styles.cardHeader}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
              <h2>Top Recommendations</h2>
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '0.9rem', color: '#8b949e' }}>
                <span>Capital: ¥</span>
                <input
                  type="number"
                  value={capital}
                  onChange={(e) => setCapital(Number(e.target.value))}
                  style={{
                    background: '#0d1117', border: '1px solid #30363d', color: '#c9d1d9',
                    padding: '4px 8px', borderRadius: '4px', width: '100px'
                  }}
                />
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
              <span className={styles.date}>{recs?.date || ''}</span>
              {recs?.generation_time && (
                <span style={{ fontSize: '0.75rem', color: '#58a6ff', opacity: 0.8 }}>
                  Generated: {recs.generation_time}
                </span>
              )}
            </div>
          </div>
          <div className={styles.list}>
            {!recError && recs?.top_recommendations?.length > 0 && (
              <div className={styles.listHeader}>
                <div className={styles.rank} title="Rank based on Model Score">#</div>
                <div className={styles.symbol} title="ETF Ticker Symbol">ETF</div>
                <div style={{ flex: 1.5, fontWeight: 500, color: '#c9d1d9', fontSize: '0.9rem' }} title="ETF Name">Name</div>
                <div className={styles.allocation} title="Recommended Weight (Risk Parity)">Alloc</div>
                <div className={styles.score} title="Raw Model Prediction Score">Score</div>
                <div style={{ marginLeft: '1rem', width: '60px', fontSize: '0.85rem', color: '#a0a0a0' }} title="20-Day Annualized Volatility">Vol</div>
                <div style={{ marginLeft: 'auto', marginRight: '1rem', fontSize: '0.85rem', color: '#a0a0a0', width: '80px', textAlign: 'right' }} title="Close Price">Price</div>
                <div style={{ width: '80px', fontSize: '0.85rem', color: '#e6edf3', textAlign: 'right', fontWeight: 600 }} title="Estimated Shares to Buy">Shares</div>
              </div>
            )}

            {recError ? <div className={styles.placeholder}>Error loading data</div> : null}
            {!recError && recs?.top_recommendations?.length > 0 ? (
              recs.top_recommendations.map((item, i) => {
                const price = item.current_price || 0;
                const topk = recs.strategy_config?.topk || 5;
                // For buffer items (Rank 6/7), assume equal weight (1/K) for share estimation
                const weight = item.target_weight > 0 ? item.target_weight : (item.is_buffer ? (1.0 / topk) : 0);

                const rawShares = (price > 0 && weight > 0)
                  ? (capital * weight) / price
                  : 0;
                const shares = Math.floor(rawShares / 100) * 100;

                return (
                  <div key={item.symbol} className={`${styles.listItem} ${item.is_buffer ? styles.bufferItem : ''}`}>
                    <div className={`${styles.rank} ${item.is_buffer ? styles.rankBuffer : ''}`}>
                      #{item.rank || i + 1}
                    </div>
                    <div className={styles.symbol}>{item.symbol}</div>
                    <div style={{ flex: 1.5, color: '#8b949e', fontSize: '0.85rem' }}>{item.name || '-'}</div>

                    <div className={styles.allocation}>
                      {item.target_weight ? `${(item.target_weight * 100).toFixed(1)}%` : '-'}
                    </div>

                    <div className={styles.score}>{item?.score?.toFixed(4)}</div>
                    {item.volatility !== undefined && (
                      <div className={styles.volatility} title="20-Day Volatility">
                        v{(item.volatility * 100).toFixed(2)}%
                      </div>
                    )}

                    {/* Price & Shares */}
                    <div style={{ marginLeft: 'auto', marginRight: '1rem', width: '80px', textAlign: 'right', color: '#8b949e', fontSize: '0.9rem' }}>
                      {price > 0 ? `¥${price.toFixed(3)}` : '-'}
                    </div>
                    <div style={{ width: '80px', textAlign: 'right', color: '#7ee787', fontWeight: 'bold', fontSize: '0.9rem' }}>
                      {shares > 0 ? shares.toLocaleString() : '-'}
                    </div>
                  </div>
                );
              })
            ) : (
              <div className={styles.placeholder}>No recommendations. Run daily task.</div>
            )}
          </div>
        </div>

        {/* Row 2: Allocation Chart (Right) */}
        <div className={`${styles.card} ${styles.col4}`}>
          <div className={styles.cardHeader}>
            <h2>Portfolio Allocation</h2>
          </div>
          {recs?.top_recommendations ? (
            <AllocationChart recommendations={recs.top_recommendations} />
          ) : (
            <div className={styles.placeholder}>No data</div>
          )}
        </div>
      </div>
    </>
  );
}
