import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
        return (
            <div style={{
                background: 'rgba(22, 27, 34, 0.95)',
                border: '1px solid #30363d',
                padding: '10px 14px',
                borderRadius: '6px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
            }}>
                <p style={{ margin: 0, marginBottom: '4px', color: '#8b949e', fontSize: '0.85rem' }}>{label}</p>
                {payload.map((entry, index) => (
                    <p key={index} style={{ margin: 0, color: entry.color, fontWeight: 600 }}>
                        {entry.name}: {((entry.value - 1) * 100).toFixed(2)}%
                    </p>
                ))}
            </div>
        );
    }
    return null;
};

export default function PerformanceChart({ data }) {
    if (!data || data.length === 0) {
        return (
            <div style={{
                height: '300px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#8b949e',
                fontStyle: 'italic'
            }}>
                Run a backtest to see performance
            </div>
        );
    }

    return (
        <div style={{ width: '100%', height: 300 }}>
            <ResponsiveContainer>
                <LineChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                    <XAxis
                        dataKey="date"
                        tick={{ fill: '#8b949e', fontSize: 11 }}
                        tickFormatter={(val) => val.slice(5)} // Show MM-DD
                        interval="preserveStartEnd"
                    />
                    <YAxis
                        tick={{ fill: '#8b949e', fontSize: 11 }}
                        tickFormatter={(val) => `${((val - 1) * 100).toFixed(0)}%`}
                        domain={['dataMin', 'dataMax']}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend
                        wrapperStyle={{ paddingTop: '10px' }}
                        iconType="line"
                    />
                    <Line
                        type="monotone"
                        dataKey="strategy"
                        name="Strategy"
                        stroke="#58a6ff"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4, fill: '#58a6ff' }}
                    />
                    <Line
                        type="monotone"
                        dataKey="benchmark"
                        name="Benchmark"
                        stroke="#8b949e"
                        strokeWidth={1.5}
                        strokeDasharray="5 5"
                        dot={false}
                        activeDot={{ r: 4, fill: '#8b949e' }}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}
