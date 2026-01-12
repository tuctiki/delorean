import { useState, useEffect } from 'react';
import useSWR from 'swr';
import Layout from '../../components/Layout';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import styles from '../../styles/Home.module.css';
import { ArrowLeft } from 'lucide-react';
import Link from 'next/link';

const fetcher = (...args) => fetch(...args).then(res => res.json());

export default function FilteredPerformance() {
    const { data: results, error } = useSWR('http://localhost:8000/api/experiment_results', fetcher);

    if (error) return <Layout><div>Error loading experiment results.</div></Layout>;
    if (!results) return <Layout><div>Loading...</div></Layout>;

    return (
        <Layout>
            <div className={styles.grid}>
                <div style={{ gridColumn: '1 / -1', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <Link href="/">
                        <ArrowLeft size={24} style={{ cursor: 'pointer', color: '#c9d1d9' }} />
                    </Link>
                    <h1 className={styles.title}>Experimental Results</h1>
                </div>

                <div className={`${styles.card} ${styles.col12}`}>
                    <div className={styles.cardHeader}>
                        <div>
                            <h2>{results.description || "Experiment Analysis"}</h2>
                            <div style={{ fontSize: '0.9rem', color: '#8b949e', marginTop: '5px' }}>
                                Period: {results.period}
                            </div>
                        </div>
                    </div>

                    <div className={styles.stats} style={{ marginTop: '20px', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px' }}>
                        <div className={styles.statRow} style={{ flexDirection: 'column', alignItems: 'flex-start', padding: '15px', background: '#161b22', borderRadius: '6px' }}>
                            <span style={{ marginBottom: '5px', fontSize: '1rem' }}>Sharpe Ratio</span>
                            <strong style={{ fontSize: '1.5rem', color: '#58a6ff' }}>{results.sharpe?.toFixed(4)}</strong>
                        </div>
                        <div className={styles.statRow} style={{ flexDirection: 'column', alignItems: 'flex-start', padding: '15px', background: '#161b22', borderRadius: '6px' }}>
                            <span style={{ marginBottom: '5px', fontSize: '1rem' }}>Ann. Return</span>
                            <strong style={{ fontSize: '1.5rem', color: results.annualized_return > 0 ? '#2ecc71' : '#e74c3c' }}>
                                {(results.annualized_return * 100).toFixed(2)}%
                            </strong>
                        </div>
                        <div className={styles.statRow} style={{ flexDirection: 'column', alignItems: 'flex-start', padding: '15px', background: '#161b22', borderRadius: '6px' }}>
                            <span style={{ marginBottom: '5px', fontSize: '1rem' }}>Max Drawdown</span>
                            <strong style={{ fontSize: '1.5rem', color: '#e74c3c' }}>
                                {(results.max_drawdown * 100).toFixed(2)}%
                            </strong>
                        </div>
                        <div className={styles.statRow} style={{ flexDirection: 'column', alignItems: 'flex-start', padding: '15px', background: '#161b22', borderRadius: '6px' }}>
                            <span style={{ marginBottom: '5px', fontSize: '1rem' }}>Win Rate</span>
                            <strong style={{ fontSize: '1.5rem', color: '#e6edf3' }}>
                                {(results.win_rate * 100).toFixed(2)}%
                            </strong>
                        </div>
                    </div>

                    <div style={{ height: '400px', marginTop: '30px', width: '100%' }}>
                        <h3 style={{ marginBottom: '15px', color: '#c9d1d9' }}>Cumulative Returns</h3>
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={results.chart_data}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                                <XAxis
                                    dataKey="date"
                                    stroke="#8b949e"
                                    tick={{ fill: '#8b949e', fontSize: 12 }}
                                    tickFormatter={(str) => str.substring(0, 4)}
                                    minTickGap={30}
                                />
                                <YAxis
                                    stroke="#8b949e"
                                    tick={{ fill: '#8b949e', fontSize: 12 }}
                                    domain={['auto', 'auto']}
                                />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#0d1117', border: '1px solid #30363d', color: '#c9d1d9' }}
                                    labelStyle={{ color: '#8b949e' }}
                                />
                                <Legend />
                                <Line
                                    type="monotone"
                                    dataKey="strategy"
                                    name="Strategy (Per-Asset Trend)"
                                    stroke="#58a6ff"
                                    dot={false}
                                    strokeWidth={2}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="benchmark"
                                    name="Benchmark (HS300)"
                                    stroke="#8b949e"
                                    dot={false}
                                    strokeWidth={1}
                                    strokeDasharray="5 5"
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>
        </Layout>
    );
}
