class MovieLensApp {
    constructor() {
        this.interactions = [];
        this.items = new Map();
        this.userMap = new Map();
        this.itemMap = new Map();
        this.userReverseMap = new Map();
        this.itemReverseMap = new Map();
        this.userRatedItems = new Map();
        this.userTopRated = new Map();
        
        this.model = null;
        this.lossHistory = [];
        
        this.initializeUI();
    }

    initializeUI() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.train());
        document.getElementById('test').addEventListener('click', () => this.test());
    }

    async loadData() {
        this.updateStatus('Loading u.data...');
        try {
            // Load interactions
            const dataResponse = await fetch('data/u.data');
            const dataText = await dataResponse.text();
            const dataLines = dataText.trim().split('\n');
            
            this.interactions = dataLines.map(line => {
                const [userId, itemId, rating, timestamp] = line.split('\t');
                return {
                    userId: parseInt(userId),
                    itemId: parseInt(itemId),
                    rating: parseFloat(rating),
                    timestamp: parseInt(timestamp)
                };
            });

            this.updateStatus('Loading u.item...');
            // Load items
            const itemResponse = await fetch('data/u.item');
            const itemText = await itemResponse.text();
            const itemLines = itemText.trim().split('\n');
            
            itemLines.forEach(line => {
                const parts = line.split('|');
                const itemId = parseInt(parts[0]);
                const title = parts[1];
                const yearMatch = title.match(/\((\d{4})\)$/);
                const year = yearMatch ? parseInt(yearMatch[1]) : 0;
                const genres = parts.slice(5, 24).map(g => parseInt(g));
                
                this.items.set(itemId, {
                    title: title.replace(/\(\d{4}\)$/, '').trim(),
                    year: year,
                    genres: genres
                });
            });

            this.updateStatus('Building indices...');
            this.buildIndices();
            this.analyzeUserBehavior();
            
            this.updateStatus(`Loaded ${this.interactions.length} interactions, ${this.items.size} items, ${this.userMap.size} users`);
        } catch (error) {
            this.updateStatus(`Error loading data: ${error.message}`);
        }
    }

    buildIndices() {
        // Build user and item mappings to 0-based indices
        const uniqueUsers = [...new Set(this.interactions.map(i => i.userId))].sort((a, b) => a - b);
        const uniqueItems = [...new Set(this.interactions.map(i => i.itemId))].sort((a, b) => a - b);
        
        uniqueUsers.forEach((userId, index) => {
            this.userMap.set(userId, index);
            this.userReverseMap.set(index, userId);
        });
        
        uniqueItems.forEach((itemId, index) => {
            this.itemMap.set(itemId, index);
            this.itemReverseMap.set(index, itemId);
        });
    }

    analyzeUserBehavior() {
        // Build user -> rated items mapping and compute top rated
        this.interactions.forEach(interaction => {
            const userId = interaction.userId;
            if (!this.userRatedItems.has(userId)) {
                this.userRatedItems.set(userId, []);
            }
            this.userRatedItems.get(userId).push(interaction);
        });

        // Compute top-rated movies per user (by rating, then recency)
        this.userRatedItems.forEach((interactions, userId) => {
            const sorted = interactions.sort((a, b) => {
                if (b.rating !== a.rating) return b.rating - a.rating;
                return b.timestamp - a.timestamp;
            });
            this.userTopRated.set(userId, sorted.slice(0, 10));
        });
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
        console.log(message);
    }

    async train() {
        const modelType = document.getElementById('modelType').value;
        this.updateStatus(`Initializing ${modelType} model...`);
        
        const config = {
            numUsers: this.userMap.size,
            numItems: this.itemMap.size,
            embeddingDim: 32,
            learningRate: 0.001,
            batchSize: 512,
            epochs: 20,
            maxInteractions: 80000
        };

        // Initialize appropriate model
        if (modelType === 'baseline') {
            this.model = new BaselineTwoTowerModel(config);
        } else {
            this.model = new DeepTwoTowerModel(config);
        }

        // Prepare training data with limited interactions
        const trainingInteractions = this.interactions
            .slice(0, config.maxInteractions)
            .map(interaction => ({
                userId: this.userMap.get(interaction.userId),
                itemId: this.itemMap.get(interaction.itemId),
                rating: interaction.rating
            }));

        this.updateStatus(`Training ${modelType} model on ${trainingInteractions.length} interactions...`);
        this.lossHistory = [];

        // Initialize loss chart
        this.initializeLossChart();

        // Training loop
        for (let epoch = 0; epoch < config.epochs; epoch++) {
            this.updateStatus(`Epoch ${epoch + 1}/${config.epochs}`);
            
            // Shuffle and batch data
            tf.util.shuffle(trainingInteractions);
            const batches = this.createBatches(trainingInteractions, config.batchSize);
            
            let epochLoss = 0;
            for (let i = 0; i < batches.length; i++) {
                const batch = batches[i];
                const loss = await this.model.trainStep(batch);
                epochLoss += loss;
                
                this.lossHistory.push(loss);
                this.updateLossChart();
                
                if (i % 10 === 0) {
                    this.updateStatus(`Epoch ${epoch + 1}, Batch ${i}/${batches.length}, Loss: ${loss.toFixed(4)}`);
                }
                
                // Clean up tensors
                tf.dispose(batch);
            }
            
            this.updateStatus(`Epoch ${epoch + 1} completed. Average loss: ${(epochLoss / batches.length).toFixed(4)}`);
        }

        this.updateStatus('Training completed! Generating embedding visualization...');
        await this.generateEmbeddingProjection();
    }

    createBatches(interactions, batchSize) {
        const batches = [];
        for (let i = 0; i < interactions.length; i += batchSize) {
            const batch = interactions.slice(i, i + batchSize);
            
            const userIds = batch.map(interaction => interaction.userId);
            const itemIds = batch.map(interaction => interaction.itemId);
            
            batches.push({
                userIds: tf.tensor1d(userIds, 'int32'),
                itemIds: tf.tensor1d(itemIds, 'int32')
            });
        }
        return batches;
    }

    initializeLossChart() {
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.fillText('Loss will appear here during training...', 20, 150);
    }

    updateLossChart() {
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        if (this.lossHistory.length === 0) return;
        
        // Draw axes
        ctx.strokeStyle = '#ccc';
        ctx.beginPath();
        ctx.moveTo(50, 20);
        ctx.lineTo(50, height - 30);
        ctx.lineTo(width - 20, height - 30);
        ctx.stroke();
        
        // Calculate scaling
        const maxLoss = Math.max(...this.lossHistory.slice(-1000));
        const minLoss = Math.min(...this.lossHistory.slice(-1000));
        const lossRange = maxLoss - minLoss || 1;
        
        // Draw loss curve (last 1000 points)
        const pointsToShow = Math.min(1000, this.lossHistory.length);
        const startIdx = this.lossHistory.length - pointsToShow;
        
        ctx.strokeStyle = '#e74c3c';
        ctx.beginPath();
        
        for (let i = 0; i < pointsToShow; i++) {
            const x = 50 + (i / pointsToShow) * (width - 70);
            const loss = this.lossHistory[startIdx + i];
            const y = height - 30 - ((loss - minLoss) / lossRange) * (height - 50);
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
        
        // Add labels
        ctx.fillStyle = '#333';
        ctx.fillText('Loss', 10, height / 2);
        ctx.fillText('Batch', width / 2, height - 10);
        ctx.fillText(`Current: ${this.lossHistory[this.lossHistory.length - 1].toFixed(4)}`, 60, 20);
    }

    async generateEmbeddingProjection() {
        this.updateStatus('Generating PCA projection...');
        
        // Sample items for visualization
        const sampleSize = Math.min(1000, this.itemMap.size);
        const sampleIndices = Array.from({length: sampleSize}, (_, i) => i);
        
        // Get item embeddings from model
        const itemEmbeddings = await this.model.getItemEmbeddings(sampleIndices);
        
        // Simple PCA implementation for 2D projection
        const projection = this.computePCA(itemEmbeddings, 2);
        
        this.drawProjection(projection, sampleIndices);
    }

    computePCA(data, components) {
        // Center the data
        const mean = data.reduce((sum, row) => row.map((val, i) => val + sum[i]), 
                                new Array(data[0].length).fill(0))
                        .map(val => val / data.length);
        
        const centered = data.map(row => row.map((val, i) => val - mean[i]));
        
        // Compute covariance matrix
        const cov = [];
        for (let i = 0; i < data[0].length; i++) {
            cov[i] = [];
            for (let j = 0; j < data[0].length; j++) {
                cov[i][j] = centered.reduce((sum, row) => sum + row[i] * row[j], 0) / (data.length - 1);
            }
        }
        
        // Simple power iteration for first two components
        // In production, use a proper PCA implementation
        const projected = data.map(row => [row[0], row[1]]); // Simplified
        
        return projected;
    }

    drawProjection(projection, indices) {
        const canvas = document.getElementById('projectionChart');
        const ctx = canvas.getContext('2d');
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Find bounds
        const xValues = projection.map(p => p[0]);
        const yValues = projection.map(p => p[1]);
        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        const yMin = Math.min(...yValues);
        const yMax = Math.max(...yValues);
        
        const scaleX = (canvas.width - 40) / (xMax - xMin);
        const scaleY = (canvas.height - 40) / (yMax - yMin);
        const scale = Math.min(scaleX, scaleY);
        
        // Draw points
        projection.forEach((point, i) => {
            const x = 20 + (point[0] - xMin) * scale;
            const y = canvas.height - 20 - (point[1] - yMin) * scale;
            
            ctx.fillStyle = '#3498db';
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
        
        ctx.fillStyle = '#333';
        ctx.fillText('PCA Projection of Item Embeddings', 20, 20);
    }

    async test() {
        if (!this.model) {
            this.updateStatus('Please train a model first!');
            return;
        }

        this.updateStatus('Finding qualified user...');
        
        // Find users with at least 20 ratings
        const qualifiedUsers = Array.from(this.userRatedItems.entries())
            .filter(([_, interactions]) => interactions.length >= 20)
            .map(([userId]) => userId);
        
        if (qualifiedUsers.length === 0) {
            this.updateStatus('No qualified users found (need users with ≥20 ratings)');
            return;
        }
        
        // Select random qualified user
        const randomUser = qualifiedUsers[Math.floor(Math.random() * qualifiedUsers.length)];
        const userIndex = this.userMap.get(randomUser);
        
        this.updateStatus(`Generating recommendations for user ${randomUser}...`);
        
        // Get user's top rated movies
        const topRated = this.userTopRated.get(randomUser) || [];
        
        // Generate recommendations from both models
        const allItemIndices = Array.from({length: this.itemMap.size}, (_, i) => i);
        const userRatedItemIndices = topRated.map(interaction => 
            this.itemMap.get(interaction.itemId)
        );
        
        // Get baseline model recommendations
        const baselineModel = new BaselineTwoTowerModel({
            numUsers: this.userMap.size,
            numItems: this.items.size,
            embeddingDim: 32
        });
        
        const baselineRecs = await this.getRecommendations(
            baselineModel, userIndex, allItemIndices, userRatedItemIndices
        );
        
        // Get deep model recommendations
        const deepRecs = await this.getRecommendations(
            this.model, userIndex, allItemIndices, userRatedItemIndices
        );
        
        this.renderComparisonTable(topRated, baselineRecs, deepRecs);
        this.updateStatus(`Recommendations generated for user ${randomUser}`);
    }

    async getRecommendations(model, userIndex, allItemIndices, excludeIndices) {
        const excludeSet = new Set(excludeIndices);
        const candidateIndices = allItemIndices.filter(idx => !excludeSet.has(idx));
        
        // Batch predictions to avoid memory issues
        const batchSize = 1000;
        const scores = [];
        
        for (let i = 0; i < candidateIndices.length; i += batchSize) {
            const batchItems = candidateIndices.slice(i, i + batchSize);
            const userTensor = tf.tensor1d(Array(batchItems.length).fill(userIndex), 'int32');
            const itemTensor = tf.tensor1d(batchItems, 'int32');
            
            const batchScores = await model.predict(userTensor, itemTensor);
            const batchScoresArray = await batchScores.array();
            
            scores.push(...batchScoresArray.map((score, idx) => ({
                itemIndex: batchItems[idx],
                score: score
            })));
            
            tf.dispose([userTensor, itemTensor, batchScores]);
        }
        
        // Return top 10 by score
        return scores.sort((a, b) => b.score - a.score)
                    .slice(0, 10)
                    .map(item => this.itemReverseMap.get(item.itemIndex));
    }

    renderComparisonTable(topRated, baselineRecs, deepRecs) {
        const container = document.getElementById('comparisonResults');
        
        let html = `
            <h3>Recommendation Comparison</h3>
            <table class="comparison-table">
                <tr>
                    <th>Top 10 Rated (Historical)</th>
                    <th>Baseline Recommendations</th>
                    <th>Deep Model Recommendations</th>
                </tr>
        `;
        
        for (let i = 0; i < 10; i++) {
            const ratedItem = topRated[i] ? this.items.get(topRated[i].itemId) : null;
            const baselineItem = baselineRecs[i] ? this.items.get(baselineRecs[i]) : null;
            const deepItem = deepRecs[i] ? this.items.get(deepRecs[i]) : null;
            
            html += `
                <tr>
                    <td>${ratedItem ? `${ratedItem.title} (${ratedItem.year}) - Rating: ${topRated[i].rating}` : ''}</td>
                    <td>${baselineItem ? `${baselineItem.title} (${baselineItem.year})` : ''}</td>
                    <td>${deepItem ? `${deepItem.title} (${deepItem.year})` : ''}</td>
                </tr>
            `;
        }
        
        html += '</table>';
        container.innerHTML = html;
    }
}

// Initialize application when loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});
