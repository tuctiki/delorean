
import useSWR from 'swr';
import Link from 'next/link';
import { Layers, FileText, TrendingUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import styles from '../../styles/Experiments.module.css';

const fetcher = (...args) => fetch(...args).then(res => res.json());

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Experiments() {
    const { data: experiments, error: expError, mutate: mutateExperiments } = useSWR(`${API_URL}/api/experiments`, fetcher);
    const { data: results, error: resultsError, mutate: mutateResults } = useSWR(`${API_URL}/api/experiment_results`, fetcher);

    return (
        <div style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>
            {/* Header Removed as per user request */}

            {/* Section 1: Latest Backtest Results */}
            <div style={{
                background: '#0d1117',
                border: '1px solid #30363d',
                borderRadius: '8px',
                padding: '20px',
                marginBottom: '30px'
            }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                    <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px', margin: 0 }}>
                        <TrendingUp size={22} color="#58a6ff" /> Latest Backtest Performance
                    </h2>
                </div>




                {results && Object.keys(results).length > 0 ? (
                    <>
                        <div style={{ fontSize: '0.9rem', color: '#8b949e', marginBottom: '20px' }}>
                            {results.description || 'Strategy Performance'} • Period: {results.period || 'N/A'}
                        </div>

                        {/* Metrics Grid */}
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '15px', marginBottom: '25px' }}>
                            <div style={{ background: '#161b22', padding: '15px', borderRadius: '6px', textAlign: 'center' }}>
                                <div style={{ fontSize: '0.85rem', color: '#8b949e', marginBottom: '5px' }}>Sharpe Ratio</div>
                                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#58a6ff' }}>
                                    {results.sharpe?.toFixed(2) || '-'}
                                </div>
                            </div>
                            <div style={{ background: '#161b22', padding: '15px', borderRadius: '6px', textAlign: 'center' }}>
                                <div style={{ fontSize: '0.85rem', color: '#8b949e', marginBottom: '5px' }}>Ann. Return</div>
                                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: results.annualized_return > 0 ? '#2ecc71' : '#e74c3c' }}>
                                    {results.annualized_return ? `${(results.annualized_return * 100).toFixed(1)}%` : '-'}
                                </div>
                            </div>
                            <div style={{ background: '#161b22', padding: '15px', borderRadius: '6px', textAlign: 'center' }}>
                                <div style={{ fontSize: '0.85rem', color: '#8b949e', marginBottom: '5px' }}>Max Drawdown</div>
                                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#e74c3c' }}>
                                    {results.max_drawdown ? `${(results.max_drawdown * 100).toFixed(1)}%` : '-'}
                                </div>
                            </div>
                            <div style={{ background: '#161b22', padding: '15px', borderRadius: '6px', textAlign: 'center' }}>
                                <div style={{ fontSize: '0.85rem', color: '#8b949e', marginBottom: '5px' }}>Win Rate</div>
                                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#c9d1d9' }}>
                                    {results.win_rate ? `${(results.win_rate * 100).toFixed(1)}%` : '-'}
                                </div>
                            </div>
                            <div style={{ background: '#161b22', padding: '15px', borderRadius: '6px', textAlign: 'center' }}>
                                <div style={{ fontSize: '0.85rem', color: '#8b949e', marginBottom: '5px' }}>Ann. Turnover</div>
                                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f1c40f' }}>
                                    {results.annualized_turnover ? `${(results.annualized_turnover * 100).toFixed(0)}%` : '-'}
                                </div>
                                <div style={{ fontSize: '0.7rem', color: '#8b949e', marginTop: '3px' }}>
                                    {results.trading_days && results.total_days ? `${results.trading_days}/${results.total_days} days` : ''}
                                </div>
                            </div>
                        </div>

                        {/* Chart */}
                        {results.chart_data && results.chart_data.length > 0 && (
                            <div style={{ height: '350px', width: '100%' }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={results.chart_data}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                                        <XAxis
                                            dataKey="date"
                                            stroke="#8b949e"
                                            tick={{ fill: '#8b949e', fontSize: 11 }}
                                            tickFormatter={(str) => str?.substring(0, 7) || ''}
                                            minTickGap={50}
                                        />
                                        <YAxis
                                            stroke="#8b949e"
                                            tick={{ fill: '#8b949e', fontSize: 11 }}
                                            domain={['auto', 'auto']}
                                        />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#0d1117', border: '1px solid #30363d', color: '#c9d1d9' }}
                                            labelStyle={{ color: '#8b949e' }}
                                        />
                                        <Legend />
                                        <Line type="monotone" dataKey="strategy" name="Strategy" stroke="#58a6ff" dot={false} strokeWidth={2} />
                                        <Line type="monotone" dataKey="benchmark" name="Benchmark (HS300)" stroke="#8b949e" dot={false} strokeWidth={1} strokeDasharray="5 5" />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        )}
                    </>
                ) : (
                    <div style={{ padding: '40px', textAlign: 'center', color: '#8b949e' }}>
                        {resultsError ? 'Error loading backtest results' : (
                            <span>
                                No backtest results available. <Link href="/operations" style={{ color: '#58a6ff' }}>Run a backtest in Operations</Link> to see performance.
                            </span>
                        )}
                    </div>
                )}
            </div>

            {/* Section 2: Experiment History */}
            <div style={{
                background: '#0d1117',
                border: '1px solid #30363d',
                borderRadius: '8px',
                padding: '20px'
            }}>
                <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
                    <Layers size={22} color="#d2a8ff" /> Run History
                </h2>

                {expError ? (
                    <div style={{ padding: '20px', textAlign: 'center', color: '#e74c3c' }}>Error loading experiments</div>
                ) : !experiments ? (
                    <div style={{ padding: '20px', textAlign: 'center', color: '#8b949e' }}>Loading...</div>
                ) : experiments.length === 0 ? (
                    <div style={{ padding: '20px', textAlign: 'center', color: '#8b949e' }}>No runs found. Run a backtest to start tracking.</div>
                ) : (
                    <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                            <thead>
                                <tr style={{ color: '#8b949e', textAlign: 'left', borderBottom: '1px solid #30363d' }}>
                                    <th style={{ padding: '12px 8px' }}>Run ID</th>
                                    <th style={{ padding: '12px 8px' }}>Name</th>
                                    <th style={{ padding: '12px 8px', textAlign: 'right' }}>Sharpe</th>
                                    <th style={{ padding: '12px 8px', textAlign: 'right' }}>Rank IC</th>
                                    <th style={{ padding: '12px 8px', textAlign: 'right' }}>Turnover</th>
                                    <th style={{ padding: '12px 8px' }}>Created</th>
                                    <th style={{ padding: '12px 8px', textAlign: 'right' }}>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {experiments.map((exp) => (
                                    <tr key={exp.id} style={{ borderBottom: '1px solid #21262d' }}>
                                        <td style={{ padding: '12px 8px', color: '#58a6ff', fontWeight: 600 }}>#{exp.id}</td>
                                        <td style={{ padding: '12px 8px', color: '#c9d1d9' }}>{exp.name}</td>
                                        <td style={{ padding: '12px 8px', textAlign: 'right', fontFamily: 'monospace', color: (exp.metrics?.sharpe || 0) > 0 ? '#2ecc71' : '#e74c3c' }}>
                                            {typeof exp.metrics?.sharpe === 'number' ? exp.metrics.sharpe.toFixed(3) : '-'}
                                        </td>
                                        <td style={{ padding: '12px 8px', textAlign: 'right', fontFamily: 'monospace', color: (exp.metrics?.rank_ic || 0) > 0.02 ? '#2ecc71' : '#f1c40f' }}>
                                            {typeof exp.metrics?.rank_ic === 'number' ? exp.metrics.rank_ic.toFixed(4) : '-'}
                                        </td>
                                        <td style={{ padding: '12px 8px', textAlign: 'right', fontFamily: 'monospace', color: '#f1c40f' }}>
                                            {typeof exp.metrics?.ann_turnover === 'number' ? `${(exp.metrics.ann_turnover * 100).toFixed(0)}%` : '-'}
                                        </td>
                                        <td style={{ padding: '12px 8px', color: '#8b949e', fontSize: '0.85rem' }}>
                                            {exp.timestamp ? new Date(exp.timestamp * 1000).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false }) : '-'}
                                        </td>
                                        <td style={{ padding: '12px 8px', textAlign: 'right' }}>
                                            <Link href={`/experiments/${exp.id}`} style={{
                                                color: '#58a6ff',
                                                textDecoration: 'none',
                                                display: 'inline-flex',
                                                alignItems: 'center',
                                                gap: '5px',
                                                padding: '4px 10px',
                                                border: '1px solid #30363d',
                                                borderRadius: '4px',
                                                fontSize: '0.85rem'
                                            }}>
                                                <FileText size={14} /> Details
                                            </Link>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
}
