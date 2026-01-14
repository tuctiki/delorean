import { useState } from 'react';
import useSWR from 'swr';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { Search } from 'lucide-react';
import styles from '../../styles/Data.module.css';

const fetcher = (...args) => fetch(...args).then(res => res.json());

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function DataViewer() {
    const { data: etfs } = useSWR(`${API_URL}/api/search`, fetcher);
    const [selectedSymbol, setSelectedSymbol] = useState(null);
    const [searchTerm, setSearchTerm] = useState('');

    const { data: history } = useSWR(selectedSymbol ? `${API_URL}/api/data/${selectedSymbol}` : null, fetcher);

    const filteredEtfs = etfs?.filter(s => s.toLowerCase().includes(searchTerm.toLowerCase()));

    return (
        <div className={styles.container}>
            <div className={styles.sidebar}>
                <div className={styles.searchBox}>
                    <Search size={18} />
                    <input
                        type="text"
                        placeholder="Search ETF..."
                        className={styles.input}
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                    />
                </div>
                <div className={styles.etfList}>
                    {filteredEtfs?.map(symbol => (
                        <div
                            key={symbol}
                            className={`${styles.etfItem} ${selectedSymbol === symbol ? styles.active : ''}`}
                            onClick={() => setSelectedSymbol(symbol)}
                        >
                            {symbol}
                        </div>
                    ))}
                </div>
            </div>

            <div className={styles.content}>
                {!selectedSymbol && <div className={styles.placeholder}>Select an ETF to view data</div>}

                {selectedSymbol && (
                    <>
                        <h2 className={styles.chartTitle}>{selectedSymbol} History</h2>
                        {history ? (
                            <div className={styles.chartContainer}>
                                <ResponsiveContainer width="100%" height={400}>
                                    <AreaChart data={history}>
                                        <defs>
                                            <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#58a6ff" stopOpacity={0.3} />
                                                <stop offset="95%" stopColor="#58a6ff" stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
                                        <XAxis dataKey="date" stroke="#8b949e" tick={{ fontSize: 12 }} tickMargin={10} minTickGap={50} />
                                        <YAxis stroke="#8b949e" domain={['auto', 'auto']} tick={{ fontSize: 12 }} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d', color: '#f0f6fc' }}
                                            itemStyle={{ color: '#58a6ff' }}
                                            labelStyle={{ color: '#8b949e' }}
                                        />
                                        <Area type="monotone" dataKey="close" stroke="#58a6ff" fillOpacity={1} fill="url(#colorClose)" />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>
                        ) : (
                            <div className={styles.loading}>Loading data...</div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}
