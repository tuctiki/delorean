import React from 'react';
import styles from '../styles/Skeleton.module.css';

/**
 * Skeleton loading placeholder component.
 * Provides shimmer animation while content is loading.
 */
export function SkeletonText({ width = '100%', height = '1em' }) {
    return (
        <div
            className={`${styles.skeleton} ${styles.skeletonText}`}
            style={{ width, height }}
        />
    );
}

export function SkeletonCard({ height = '200px' }) {
    return (
        <div
            className={`${styles.skeleton} ${styles.skeletonCard}`}
            style={{ height }}
        />
    );
}

export function SkeletonRow({ count = 3 }) {
    return (
        <div className={styles.skeletonRow}>
            {[...Array(count)].map((_, i) => (
                <SkeletonText key={i} width={`${Math.random() * 30 + 20}%`} />
            ))}
        </div>
    );
}

export function SkeletonList({ rows = 5 }) {
    return (
        <div>
            {[...Array(rows)].map((_, i) => (
                <SkeletonRow key={i} count={4} />
            ))}
        </div>
    );
}

export default { SkeletonText, SkeletonCard, SkeletonRow, SkeletonList };
