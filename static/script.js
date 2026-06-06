document.addEventListener('DOMContentLoaded', async function() {
    const data = await getStockData();
    const currentPrice = data[data.length - 1].Close;
    const predictionData = await fetchStockprediction();
    const pred_processed = calculatePrice(predictionData, currentPrice);
    buildscatterplot(data, predictionData);
    updateinfo(pred_processed);
});

function calculatePrice(predictionData, currentPrice) {
    const pred_returns_3d  = predictionData.move_3d  * predictionData.dir_3d;
    const pred_returns_5d  = predictionData.move_5d  * predictionData.dir_5d;
    const pred_returns_10d = predictionData.move_10d * predictionData.dir_10d;

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
    const ctx = document.getElementById('predChart');

    new Chart(ctx, {
        type: 'scatter',
        data: {
            labels: [],
            datasets: []
        },
        options: {}
    });
}

function updateinfo(pred_processed) {
    document.getElementById('current_price').textContent = `Current Price: ${pred_processed['3d'][0].toFixed(2)}`;
    document.getElementById('pred_3d').textContent  = `3d: ${pred_processed['3d'][0].toFixed(2)} (${(pred_processed['3d'][1] * 100).toFixed(2)}%)`;
    document.getElementById('pred_5d').textContent  = `5d: ${pred_processed['5d'][0].toFixed(2)} (${(pred_processed['5d'][1] * 100).toFixed(2)}%)`;
    document.getElementById('pred_10d').textContent = `10d: ${pred_processed['10d'][0].toFixed(2)} (${(pred_processed['10d'][1] * 100).toFixed(2)}%)`;
}