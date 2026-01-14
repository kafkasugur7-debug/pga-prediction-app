// ============================================
// DOM Elements
// ============================================
const form = document.getElementById('predictionForm');
const submitBtn = document.getElementById('submitBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const errorMessage = document.getElementById('errorMessage');
const resultsSection = document.getElementById('resultsSection');

// Input summary elements
const summaryVs30 = document.getElementById('summaryVs30');
const summaryRepi = document.getElementById('summaryRepi');
const summaryMw = document.getElementById('summaryMw');

// PGA Max stats
const maxMean = document.getElementById('maxMean');
const maxMax = document.getElementById('maxMax');
const maxMin = document.getElementById('maxMin');

// PGA Avg stats
const avgMean = document.getElementById('avgMean');
const avgMax = document.getElementById('avgMax');
const avgMin = document.getElementById('avgMin');

// Charts
let maxChart = null;
let avgChart = null;

// ============================================
// Form Submission Handler
// ============================================
form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Get form values
    const formData = {
        Vs30: parseFloat(document.getElementById('vs30').value),
        Repi: parseFloat(document.getElementById('repi').value),
        Mw: parseFloat(document.getElementById('mw').value)
    };

    // Hide error message and results
    errorMessage.style.display = 'none';
    resultsSection.style.display = 'none';

    // Show loading overlay
    loadingOverlay.style.display = 'flex';

    try {
        // Make API request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Tahmin yapılırken bir hata oluştu');
        }

        // Display results
        displayResults(data);

        // Smooth scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 300);

    } catch (error) {
        // Show error message
        errorMessage.textContent = `❌ ${error.message}`;
        errorMessage.style.display = 'block';
        console.error('Prediction error:', error);
    } finally {
        // Hide loading overlay
        loadingOverlay.style.display = 'none';
    }
});

// ============================================
// Display Results Function
// ============================================
function displayResults(data) {
    // Update input summary
    summaryVs30.textContent = `${data.inputs.Vs30.toFixed(1)} m/s`;
    summaryRepi.textContent = `${data.inputs.Repi.toFixed(1)} km`;
    summaryMw.textContent = data.inputs.Mw.toFixed(1);

    // Update PGA Max statistics
    maxMean.textContent = data.statistics.pga_max.mean.toFixed(2);
    maxMax.textContent = data.statistics.pga_max.max.toFixed(2);
    maxMin.textContent = data.statistics.pga_max.min.toFixed(2);

    // Update PGA Average statistics
    avgMean.textContent = data.statistics.pga_avg.mean.toFixed(2);
    avgMax.textContent = data.statistics.pga_avg.max.toFixed(2);
    avgMin.textContent = data.statistics.pga_avg.min.toFixed(2);

    // Create charts
    createMaxChart(data.pga_max, data.statistics.pga_max);
    createAvgChart(data.pga_avg, data.statistics.pga_avg);

    // Show results section
    resultsSection.style.display = 'block';
}

// ============================================
// Create PGA Max Chart
// ============================================
function createMaxChart(predictions, stats) {
    const ctx = document.getElementById('maxChart').getContext('2d');

    // Destroy existing chart if exists
    if (maxChart) {
        maxChart.destroy();
    }

    const models = Object.keys(predictions);
    const values = Object.values(predictions);

    maxChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Decision Tree', 'Gradient Boosting', 'LightGBM'],
            datasets: [
                {
                    label: 'Model Tahminleri',
                    data: values,
                    backgroundColor: [
                        'rgba(59, 130, 246, 0.8)',
                        'rgba(249, 115, 22, 0.8)',
                        'rgba(16, 185, 129, 0.8)'
                    ],
                    borderColor: [
                        'rgba(59, 130, 246, 1)',
                        'rgba(249, 115, 22, 1)',
                        'rgba(16, 185, 129, 1)'
                    ],
                    borderWidth: 2,
                    borderRadius: 8
                },
                {
                    label: 'Ortalama',
                    data: [stats.mean, stats.mean, stats.mean],
                    type: 'line',
                    borderColor: 'rgba(239, 68, 68, 1)',
                    borderWidth: 3,
                    borderDash: [10, 5],
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                title: {
                    display: true,
                    text: 'PGA Maksimum Tahminleri',
                    font: {
                        size: 18,
                        weight: 'bold'
                    },
                    color: '#0f172a'
                },
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        font: {
                            size: 12,
                            weight: '600'
                        },
                        color: '#1e293b',
                        usePointStyle: true,
                        padding: 15
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleFont: {
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 13
                    },
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: function (context) {
                            if (context.dataset.label === 'Ortalama') {
                                return `Ortalama: ${context.parsed.y.toFixed(2)} gal`;
                            }
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)} gal`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'PGA (gal)',
                        font: {
                            size: 14,
                            weight: 'bold'
                        },
                        color: '#1e293b'
                    },
                    ticks: {
                        font: {
                            size: 12
                        },
                        color: '#64748b'
                    },
                    grid: {
                        color: 'rgba(226, 232, 240, 0.5)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'ML Modelleri',
                        font: {
                            size: 14,
                            weight: 'bold'
                        },
                        color: '#1e293b'
                    },
                    ticks: {
                        font: {
                            size: 11,
                            weight: '600'
                        },
                        color: '#1e293b'
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// ============================================
// Create PGA Average Chart
// ============================================
function createAvgChart(predictions, stats) {
    const ctx = document.getElementById('avgChart').getContext('2d');

    // Destroy existing chart if exists
    if (avgChart) {
        avgChart.destroy();
    }

    const models = Object.keys(predictions);
    const values = Object.values(predictions);

    // Create range dataset (min to max shaded area)
    avgChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Decision Tree', 'Gradient Boosting', 'LightGBM'],
            datasets: [
                {
                    label: 'Model Tahminleri',
                    data: values,
                    backgroundColor: [
                        'rgba(59, 130, 246, 0.8)',
                        'rgba(249, 115, 22, 0.8)',
                        'rgba(16, 185, 129, 0.8)'
                    ],
                    borderColor: [
                        'rgba(59, 130, 246, 1)',
                        'rgba(249, 115, 22, 1)',
                        'rgba(16, 185, 129, 1)'
                    ],
                    borderWidth: 2,
                    borderRadius: 8
                },
                {
                    label: 'Ortalama',
                    data: [stats.mean, stats.mean, stats.mean],
                    type: 'line',
                    borderColor: 'rgba(239, 68, 68, 1)',
                    borderWidth: 3,
                    borderDash: [10, 5],
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: 'Min-Max Aralığı',
                    data: [
                        { y: [stats.min, stats.max] },
                        { y: [stats.min, stats.max] },
                        { y: [stats.min, stats.max] }
                    ],
                    type: 'line',
                    backgroundColor: 'rgba(147, 197, 253, 0.2)',
                    borderColor: 'rgba(147, 197, 253, 0)',
                    fill: true,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                title: {
                    display: true,
                    text: 'PGA Ortalama Tahminleri',
                    font: {
                        size: 18,
                        weight: 'bold'
                    },
                    color: '#0f172a'
                },
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        font: {
                            size: 12,
                            weight: '600'
                        },
                        color: '#1e293b',
                        usePointStyle: true,
                        padding: 15
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleFont: {
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 13
                    },
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: function (context) {
                            if (context.dataset.label === 'Ortalama') {
                                return `Ortalama: ${context.parsed.y.toFixed(2)} gal`;
                            } else if (context.dataset.label === 'Min-Max Aralığı') {
                                return `Aralık: ${stats.min.toFixed(2)} - ${stats.max.toFixed(2)} gal`;
                            }
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)} gal`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'PGA (gal)',
                        font: {
                            size: 14,
                            weight: 'bold'
                        },
                        color: '#1e293b'
                    },
                    ticks: {
                        font: {
                            size: 12
                        },
                        color: '#64748b'
                    },
                    grid: {
                        color: 'rgba(226, 232, 240, 0.5)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'ML Modelleri',
                        font: {
                            size: 14,
                            weight: 'bold'
                        },
                        color: '#1e293b'
                    },
                    ticks: {
                        font: {
                            size: 11,
                            weight: '600'
                        },
                        color: '#1e293b'
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// ============================================
// Utility Functions
// ============================================
console.log('🌍 PGA Prediction App Initialized');
