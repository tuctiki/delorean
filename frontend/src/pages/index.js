import { useState, useEffect, useRef } from 'react';
import useSWR from 'swr';
import { Play, Activity, TrendingUp, AlertTriangle, RefreshCw, ShieldCheck, Clock, BarChart3 } from 'lucide-react';
import Link from 'next/link';
import AllocationChart from '../components/AllocationChart';
import PerformanceChart from '../components/PerformanceChart';
import { SkeletonText, SkeletonList, SkeletonCard } from '../components/Skeleton';
import styles from '../styles/Home.module.css';

const fetcher = (...args) => fetch(...args).then(res => res.json());

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const { data: recs, error: recError, isLoading: recsLoading } = useSWR(`${API_URL}/api/recommendations`, fetcher, { refreshInterval: 5000 });
  const { data: status, error: statusError } = useSWR(`${API_URL}/api/status`, fetcher, { refreshInterval: 1000 });
  const { data: performance, isLoading: perfLoading } = useSWR(`${API_URL}/api/performance`, fetcher, { refreshInterval: 30000 });
  const { data: history, isLoading: historyLoading } = useSWR(`${API_URL}/api/recommendation-history`, fetcher, { refreshInterval: 60000 });

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
      await fetch(`${API_URL}/api/run-daily`, { method: 'POST' });
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

  // Validation Colors
  const getStatusColor = (status) => {
    if (status === 'Pass') return '#2ecc71';
    if (status === 'Warning') return '#f1c40f';
    if (status === 'Critical' || status === 'Error') return '#e74c3c';
    return '#8b949e';
  };

  return (
    <>
      <div className={styles.grid}>
        {/* Row 1: Task Status, Strategy, Market, Validation */}
        <div className={`${styles.card} ${styles.col3}`}>
          <div className={styles.cardHeader}>
            <h2><Activity size={20} /> Task Status</h2>
            <Link
              href="/operations"
              style={{ color: '#58a6ff', textDecoration: 'none', fontSize: '0.85rem' }}
            >
              View Logs →
            </Link>
          </div>
          <div className={styles.stats}>
            <div className={styles.statRow}>
              <span>Daily Signal</span>
              <strong style={{ color: isRunning ? '#f1c40f' : '#2ecc71' }}>
                {isRunning ? 'Running...' : 'Ready'}
              </strong>
            </div>
            <div className={styles.statRow}>
              <span>Last Update</span>
              <strong>{recs?.generation_time?.split(' ')[1] || '-'}</strong>
            </div>
          </div>
        </div>

        <div className={`${styles.card} ${styles.col3}`}>
          <div className={styles.cardHeader}>
            <h2><RefreshCw size={20} /> Active Strategy</h2>
          </div>
          {recsLoading ? (
            <div style={{ padding: '1rem' }}>
              <SkeletonText width="60%" />
              <SkeletonText width="80%" />
              <SkeletonText width="50%" />
            </div>
          ) : (
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
          )}
        </div>

        <div className={`${styles.card} ${styles.col3}`}>
          <div className={styles.cardHeader}>
            <h2><TrendingUp size={20} /> Market Status</h2>
            {recsLoading ? (
              <SkeletonText width="60px" height="24px" />
            ) : (
              <div className={`${styles.badge} ${isMarketBull ? styles.bull : styles.bear}`}>
                {recs?.market_status || 'Unknown'}
              </div>
            )}
          </div>
          {recsLoading ? (
            <div style={{ padding: '1rem' }}><SkeletonText /><SkeletonText width="70%" /></div>
          ) : (
            <div className={styles.stats}>
              <div className={styles.statRow}>
                <span>Benchmark Close</span>
                <strong>{recs?.market_data?.benchmark_close?.toFixed(2) || '-'}</strong>
              </div>
              <div className={styles.statRow}>
                <span>Benchmark MA60</span>
                <strong>{recs?.market_data?.benchmark_ma60?.toFixed(2) || recs?.market_data?.benchmark_ma?.toFixed(2) || '-'}</strong>
              </div>
            </div>
          )}
          {!isMarketBull && recs && (
            <div className={styles.alert}>
              <AlertTriangle size={16} /> Bear Market. Cash Recommended.
            </div>
          )}
        </div>

        <div className={`${styles.card} ${styles.col3}`}>
          <div className={styles.cardHeader}>
            <h2><ShieldCheck size={20} /> Model Health</h2>
          </div>
          {recsLoading ? (
            <div style={{ padding: '1rem' }}><SkeletonText /><SkeletonText width="70%" /></div>
          ) : (
            <div className={styles.stats}>
              <div className={styles.statRow}>
                <span title="Rank IC (Last 6 Months)">Rank IC</span>
                <strong style={{ color: getStatusColor(recs?.validation?.ic_status) }}>
                  {recs?.validation?.rank_ic ? recs.validation.rank_ic.toFixed(4) : '-'}
                </strong>
              </div>
              <div className={styles.statRow}>
                <span title="Simulated Sharpe (Last 60 Days)">Sharpe (60d)</span>
                <strong style={{ color: getStatusColor(recs?.validation?.sharpe_status) }}>
                  {recs?.validation?.sharpe ? recs.validation.sharpe.toFixed(4) : '-'}
                </strong>
              </div>
            </div>
          )}
          {(recs?.validation?.ic_status === 'Critical' || recs?.validation?.sharpe_status === 'Error') && (
            <div className={styles.alert} style={{ marginTop: 'auto' }}>
              <AlertTriangle size={16} /> Model Degraded
            </div>
          )}
        </div>

        {/* Row 2: Performance Chart (NEW) */}
        <div className={`${styles.card} ${styles.col12}`}>
          <div className={styles.cardHeader}>
            <h2><BarChart3 size={20} /> Historical Performance</h2>
          </div>
          {perfLoading ? (
            <SkeletonCard height="300px" />
          ) : (
            <PerformanceChart data={performance?.chart_data} />
          )}
        </div>

        {/* Row 3: Recommendations List (Left) */}
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
            {recsLoading ? (
              <SkeletonList rows={5} />
            ) : (
              <>
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
                    const weight = item.target_weight > 0 ? item.target_weight : (item.is_buffer ? (1.0 / topk) : 0);
                    const rawShares = (price > 0 && weight > 0) ? (capital * weight) / price : 0;
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
              </>
            )}
          </div>
        </div>

        {/* Row 3: Allocation Chart (Right) */}
        <div className={`${styles.card} ${styles.col4}`}>
          <div className={styles.cardHeader}>
            <h2>Portfolio Allocation</h2>
          </div>
          {recsLoading ? (
            <SkeletonCard height="250px" />
          ) : recs?.top_recommendations ? (
            <AllocationChart recommendations={recs.top_recommendations} />
          ) : (
            <div className={styles.placeholder}>No data</div>
          )}
        </div>

        {/* Row 4: Recommendation History (NEW) */}
        <div className={`${styles.card} ${styles.col12}`}>
          <div className={styles.cardHeader}>
            <h2><Clock size={20} /> Recommendation History (Last 7 Days)</h2>
          </div>
          {historyLoading ? (
            <SkeletonList rows={3} />
          ) : history?.history?.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #30363d' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left', color: '#8b949e' }}>Rank</th>
                    {history.history.map(day => (
                      <th key={day.date} style={{ padding: '8px 12px', textAlign: 'left', color: '#8b949e' }}>
                        {day.date}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[1, 2, 3, 4, 5].map(rank => (
                    <tr key={rank} style={{ borderBottom: '1px solid #21262d' }}>
                      <td style={{ padding: '8px 12px', color: '#58a6ff', fontWeight: 600 }}>#{rank}</td>
                      {history.history.map(day => {
                        const rec = day.recommendations.find(r => r.rank === rank);
                        return (
                          <td key={day.date} style={{ padding: '8px 12px' }}>
                            {rec ? (
                              <div>
                                <div style={{ fontWeight: 500, color: '#c9d1d9' }}>{rec.name}</div>
                                <div style={{ fontSize: '0.75rem', color: '#8b949e' }}>{rec.symbol}</div>
                              </div>
                            ) : '-'}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className={styles.placeholder}>
              No history available. Run the historical recommendations script.
            </div>
          )}
        </div>
      </div>
    </>
  );
}
