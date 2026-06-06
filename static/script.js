document.addEventListener('DOMContentLoaded', async function() {
    const data = await getStockData();
    const currentPrice = data[data.length - 1].Close;
    const predictionData = await fetchStockprediction();
    const pred_processed = calculatePrice(predictionData, currentPrice);
    buildscatterplot(data, predictionData);
    updateinfo(pred_processed, currentPrice);

    document.getElementById('dashboard').classList.remove('loading');
});

function calculatePrice(predictionData, currentPrice) {
    const preds = predictionData.predictions;
    const last = arr => arr[arr.length - 1];
    const pred_returns_3d  = last(preds.move_3d)  * last(preds.dir_3d);
    const pred_returns_5d  = last(preds.move_5d)  * last(preds.dir_5d);
    const pred_returns_10d = last(preds.move_10d) * last(preds.dir_10d);

    const pred_price_3d  = currentPrice * (1 + pred_returns_3d);
    const pred_price_5d  = currentPrice * (1 + pred_returns_5d);
    const pred_price_10d = currentPrice * (1 + pred_returns_10d);

    return {
        '3d':  [pred_price_3d,  pred_returns_3d],
        '5d':  [pred_price_5d,  pred_returns_5d],
        '10d': [pred_price_10d, pred_returns_10d]
    };
}

async function fetchStockprediction() {
    const response = await fetch('/predict', { method: 'POST' });
    const data = await response.json();
    return data;
}

async function getStockData() {
    const response = await fetch('/data');
    const data = await response.json();
    return data;
}

function buildscatterplot(stockData, predictionData) {
    const DAYS = 5;
    const prices = stockData.map(d => d.Close);
    const dates  = stockData.map(d => d.Date);

    const histPrices = prices.slice(-DAYS);
    const histDates  = dates.slice(-DAYS);
    const currentPrice = histPrices[histPrices.length - 1];
    const pred = calculatePrice(predictionData, currentPrice);

    const addTradingDays = (dateStr, n) => {
        const d = new Date(dateStr);
        let added = 0;
        while (added < n) {
            d.setDate(d.getDate() + 1);
            if (d.getDay() !== 0 && d.getDay() !== 6) added++;
        }
        return d.toISOString().split('T')[0];
    };
    const lastDate = histDates[histDates.length - 1];

    const N = histPrices.length;
    const allLabels = [...histDates, addTradingDays(lastDate, 3), addTradingDays(lastDate, 5), addTradingDays(lastDate, 10)];
    const histData  = [...histPrices, null, null, null];
    const predData  = [
        ...Array(N - 1).fill(null),
        currentPrice, pred['3d'][0], pred['5d'][0], pred['10d'][0]
    ];

    if (window._chart) window._chart.destroy();

    window._chart = new Chart(document.getElementById('predChart'), {
        type: 'line',
        data: {
            labels: allLabels,
            datasets: [
                {
                    label: 'Precio histórico',
                    data: histData,
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56,189,248,.06)',
                    borderWidth: 1.5,
                    pointRadius: 3,
                    pointBackgroundColor: '#38bdf8',
                    pointBorderColor: 'transparent',
                    fill: true,
                    tension: 0.3,
                },
                {
                    label: 'Predicción IA',
                    data: predData,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245,158,11,.06)',
                    borderWidth: 1.5,
                    pointRadius: [...Array(N - 1).fill(0), 0, 5, 5, 5],
                    pointBackgroundColor: '#f59e0b',
                    pointBorderColor: 'transparent',
                    fill: true,
                    tension: 0.3,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    align: 'end',
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Inter', size: 12, weight: '500' },
                        boxWidth: 24,
                        boxHeight: 2,
                        padding: 20,
                    }
                },
                tooltip: {
                    backgroundColor: '#0f1623',
                    borderColor: 'rgba(56,189,248,.15)',
                    borderWidth: 1,
                    titleColor: '#e2e8f0',
                    titleFont: { family: 'Inter', size: 12 },
                    bodyColor: '#94a3b8',
                    bodyFont: { family: 'Inter', size: 12 },
                    padding: 12,
                    callbacks: {
                        label: ctx => {
                            const v = ctx.parsed.y;
                            return v != null ? ` ${ctx.dataset.label}: $${v.toFixed(2)}` : null;
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: '#64748b',
                        font: { family: 'Inter', size: 11 },
                        maxTicksLimit: 7,
                        maxRotation: 0,
                    },
                    grid: { color: 'rgba(255,255,255,.03)' },
                    border: { color: 'transparent' }
                },
                y: {
                    position: 'right',
                    ticks: {
                        color: '#64748b',
                        font: { family: 'Inter', size: 11 },
                        callback: v => `$${v.toFixed(0)}`,
                    },
                    grid: { color: 'rgba(255,255,255,.03)' },
                    border: { color: 'transparent' }
                }
            }
        }
    });
}

function updateinfo(pred_processed, currentPrice) {
    const setCard = (id, horizon) => {
        const el = document.getElementById(id);
        const ret = pred_processed[horizon][1];
        const sign = ret >= 0 ? '+' : '';
        el.textContent = `$${pred_processed[horizon][0].toFixed(2)} (${sign}${(ret * 100).toFixed(2)}%)`;
        el.classList.remove('up', 'down');
        el.classList.add(ret >= 0 ? 'up' : 'down');
    };
    document.getElementById('current-price').textContent = `$${currentPrice.toFixed(2)}`;
    setCard('pred-3d',  '3d');
    setCard('pred-5d',  '5d');
    setCard('pred-10d', '10d');
}