import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

const COLORS = ['#2ecc71', '#3498db', '#9b59b6', '#f1c40f', '#e67e22', '#e74c3c'];

const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
        const data = payload[0].payload;
        return (
            <div style={{
                background: 'rgba(22, 27, 34, 0.95)',
                border: '1px solid #30363d',
                padding: '10px',
                borderRadius: '6px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
            }}>
                <p style={{ margin: 0, fontWeight: 'bold', color: '#f0f6fc' }}>{data.symbol}</p>
                <p style={{ margin: 0, color: '#2ecc71', fontFamily: 'monospace' }}>
                    {(data.target_weight * 100).toFixed(1)}%
                </p>
            </div>
        );
    }
    return null;
};

export default function AllocationChart({ recommendations }) {
    // Filter only items with positive weight
    const data = recommendations
        .filter(item => item.target_weight > 0)
        .map(item => ({
            name: item.symbol,
            symbol: item.symbol,
            target_weight: item.target_weight || 0,
            value: item.target_weight || 0
        }));

    if (!data || data.length === 0) return (
        <div style={{ height: '250px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#8b949e', fontStyle: 'italic' }}>
            No allocation data
        </div>
    );

    return (
        <div style={{ width: '100%', height: 300, position: 'relative' }}>
            <ResponsiveContainer>
                <PieChart>
                    <Pie
                        data={data}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={90}
                        paddingAngle={5}
                        dataKey="value"
                        stroke="none"
                    >
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                    <Legend verticalAlign="bottom" height={36} iconType="circle" />
                </PieChart>
            </ResponsiveContainer>

            {/* Center Text overlay */}
            <div style={{
                position: 'absolute',
                top: '40%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                textAlign: 'center',
                pointerEvents: 'none'
            }}>
                <div style={{ fontSize: '0.8rem', color: '#8b949e' }}>TARGET</div>
                <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#f0f6fc' }}>100%</div>
            </div>
        </div>
    );
}
