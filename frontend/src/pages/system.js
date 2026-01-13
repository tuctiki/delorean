import useSWR from 'swr';
import Link from 'next/link';
import { Settings, Database, Cpu, List } from 'lucide-react';

const fetcher = (...args) => fetch(...args).then(res => res.json());

export default function SystemSettings() {
    const { data: config, error } = useSWR('http://localhost:8000/api/config', fetcher);

    if (error) return <div>Error loading config</div>;
    if (!config) return <div>Loading...</div>;

    return (
        <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
            <div style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Link href="/" style={{ color: '#8b949e', textDecoration: 'none' }}>Dashboard</Link>
                <span style={{ color: '#30363d' }}>/</span>
                <span style={{ color: '#c9d1d9' }}>System Settings</span>
            </div>

            <h1 style={{ marginBottom: '30px', borderBottom: '1px solid #30363d', paddingBottom: '10px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Settings size={28} /> System Settings
            </h1>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: '20px' }}>

                {/* Section 1: Model Hyperparameters */}
                <div style={{ background: '#0d1117', border: '1px solid #30363d', borderRadius: '6px', padding: '20px' }}>
                    <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '1.2rem', marginBottom: '15px' }}>
                        <Cpu size={20} color="#58a6ff" /> Model Hyperparameters (Stage 1)
                    </h2>
                    <div style={{ background: '#010409', padding: '15px', borderRadius: '6px', fontSize: '0.9rem', overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                            <thead>
                                <tr style={{ color: '#8b949e', textAlign: 'left' }}>
                                    <th style={{ padding: '8px', borderBottom: '1px solid #30363d' }}>Parameter</th>
                                    <th style={{ padding: '8px', borderBottom: '1px solid #30363d' }}>Value</th>
                                </tr>
                            </thead>
                            <tbody>
                                {Object.entries(config.model_params.stage1).map(([key, value]) => (
                                    <tr key={key} style={{ borderBottom: '1px solid #21262d' }}>
                                        <td style={{ padding: '8px', color: '#c9d1d9' }}>{key}</td>
                                        <td style={{ padding: '8px', color: '#7ee787', fontFamily: 'monospace' }}>{String(value)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    <h3 style={{ fontSize: '1rem', marginTop: '20px', marginBottom: '10px', color: '#8b949e' }}>Stage 2 (Refinement)</h3>
                    <div style={{ background: '#010409', padding: '15px', borderRadius: '6px', fontSize: '0.9rem', overflowX: 'auto' }}>
                        <pre style={{ margin: 0, color: '#c9d1d9' }}>{JSON.stringify(config.model_params.stage2, null, 2)}</pre>
                    </div>
                </div>

                {/* Section 2: Data Factors */}
                <div style={{ background: '#0d1117', border: '1px solid #30363d', borderRadius: '6px', padding: '20px' }}>
                    <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '1.2rem', marginBottom: '15px' }}>
                        <Database size={20} color="#d2a8ff" /> Alpha Factors ({config.data_factors.names.length})
                    </h2>
                    <p style={{ color: '#8b949e', fontSize: '0.9rem', marginBottom: '15px' }}>
                        Custom synthetic factors for ETF rotation strategy.
                    </p>
                    <div style={{ background: '#010409', padding: '15px', borderRadius: '6px', fontSize: '0.85rem', maxHeight: '400px', overflowY: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                            <thead>
                                <tr style={{ color: '#8b949e', textAlign: 'left' }}>
                                    <th style={{ padding: '8px', borderBottom: '1px solid #30363d', width: '30%' }}>Feature Name</th>
                                    <th style={{ padding: '8px', borderBottom: '1px solid #30363d' }}>Qlib Expression</th>
                                </tr>
                            </thead>
                            <tbody>
                                {config.data_factors.names.map((name, i) => (
                                    <tr key={name} style={{ borderBottom: '1px solid #21262d' }}>
                                        <td style={{ padding: '8px', color: '#d2a8ff', fontWeight: 500 }}>{name}</td>
                                        <td style={{ padding: '8px', color: '#c9d1d9', fontFamily: 'monospace', wordBreak: 'break-all', fontSize: '0.8rem' }}>
                                            {config.data_factors.expressions[i]}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Section 3: ETF Universe */}
                <div style={{ background: '#0d1117', border: '1px solid #30363d', borderRadius: '6px', padding: '20px', gridColumn: '1 / -1' }}>
                    <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '1.2rem', marginBottom: '15px' }}>
                        <List size={20} color="#7ee787" /> ETF Universe ({config.universe.length} ETFs)
                    </h2>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '10px' }}>
                        {config.universe.map((etf) => (
                            <div key={etf} style={{
                                background: '#010409',
                                padding: '10px 15px',
                                borderRadius: '6px',
                                border: '1px solid #21262d',
                                fontFamily: 'monospace',
                                fontSize: '0.9rem',
                                color: '#c9d1d9'
                            }}>
                                {etf}
                            </div>
                        ))}
                    </div>
                </div>

            </div>
        </div>
    );
}
