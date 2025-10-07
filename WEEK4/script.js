/**
 * Main Application Controller
 * Handles UI interactions, model training, and recommendations
 */
class MovieLensApp {
    constructor() {
        this.data = new MovieLensData();
        this.model = null;
        this.lossHistory = [];
        this.isTraining = false;
        
        this.initializeUI();
    }

    initializeUI() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.train());
        document.getElementById('test').addEventListener('click', () => this.test());
        
        // Disable train/test buttons until data is loaded
        document.getElementById('train').disabled = true;
        document.getElementById('test').disabled = true;
    }

    async loadData() {
        this.updateStatus('Loading MovieLens data...');
        try {
            const stats = await this.data.loadData();
            this.updateStatus(`Loaded ${stats.interactions} interactions, ${stats.items} movies, ${stats.users} users. ${stats.qualifiedUsers} qualified users for testing.`);
            
            // Enable train button
            document.getElementById('train').disabled = false;
        } catch (error) {
            this.updateStatus(`Error loading data: ${error.message}`);
        }
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
        console.log(message);
    }

    async train() {
        if (this.isTraining) {
            this.updateStatus('Training already in progress...');
            return;
        }

        const modelType = document.getElementById('modelType').value;
        this.updateStatus(`Initializing ${modelType} model...`);
        
        this.isTraining = true;
        document.getElementById('train').disabled = true;

        try {
            const config = {
                numUsers: this.data.userMap.size,
                numItems: this.data.items.size,
                embeddingDim: 32,
                learningRate: 0.001,
                batchSize: 512,
                epochs: 20,
                maxInteractions: 80000
            };

            // Initialize appropriate model
            this.model = modelType === 'baseline' 
                ? new BaselineTwoTowerModel(config)
                : new DeepTwoTowerModel(config);

            // Prepare training data
            const trainingInteractions = this.data.getTrainingData(config.maxInteractions);
            this.updateStatus(`Training ${modelType} model on ${trainingInteractions.length} interactions...`);
            
            this.lossHistory = [];
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
                    
                    tf.dispose(batch);
                }
                
                this.updateStatus(`Epoch ${epoch + 1} completed. Average loss: ${(epochLoss / batches.length).toFixed(4)}`);
            }

            this.updateStatus('Training completed! Generating embedding visualization...');
            await this.generateEmbeddingProjection();
            
            // Enable test button
            document.getElementById('test').disabled = false;

        } catch (error) {
            this.updateStatus(`Training error: ${error.message}`);
        } finally {
            this.isTraining = false;
            document.getElementById('train').disabled = false;
        }
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
        
        // Calculate scaling for last 1000 points
        const pointsToShow = Math.min(1000, this.lossHistory.length);
        const startIdx = this.lossHistory.length - pointsToShow;
        const visibleLosses = this.lossHistory.slice(startIdx);
        
        const maxLoss = Math.max(...visibleLosses);
        const minLoss = Math.min(...visibleLosses);
        const lossRange = maxLoss - minLoss || 1;
        
        // Draw loss curve
        ctx.strokeStyle = '#e74c3c';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        for (let i = 0; i < pointsToShow; i++) {
            const x = 50 + (i / pointsToShow) * (width - 70);
            const loss = visibleLosses[i];
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
        ctx.font = '12px Arial';
        ctx.fillText('Loss', 10, height / 2);
        ctx.fillText('Batch', width / 2, height - 10);
        ctx.fillText(`Current: ${this.lossHistory[this.lossHistory.length - 1].toFixed(4)}`, 60, 20);
        ctx.fillText(`Min: ${minLoss.toFixed(4)}`, 60, 35);
        ctx.fillText(`Max: ${maxLoss.toFixed(4)}`, 60, 50);
    }

    async generateEmbeddingProjection() {
        this.updateStatus('Generating PCA projection...');
        
        // Sample items for visualization
        const sampleSize = Math.min(500, this.data.itemMap.size);
        const sampleIndices = Array.from({length: sampleSize}, (_, i) => i);
        
        try {
            const itemEmbeddings = await this.model.getItemEmbeddings(sampleIndices);
            const projection = this.computePCA(itemEmbeddings, 2);
            this.drawProjection(projection, sampleIndices);
        } catch (error) {
            console.error('Error generating projection:', error);
            this.updateStatus('Error generating embedding visualization');
        }
    }

    computePCA(data, components) {
        // Simplified PCA implementation using first two dimensions
        // In production, use a proper PCA implementation
        const normalized = data.map(row => {
            const mean = row.reduce((sum, val) => sum + val, 0) / row.length;
            const std = Math.sqrt(row.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / row.length);
            return row.map(val => (val - mean) / (std || 1));
        });
        
        // Use first two dimensions as approximation
        return normalized.map(row => [row[0], row[1]]);
    }

    drawProjection(projection, indices) {
        const canvas = document.getElementById('projectionChart');
        const ctx = canvas.getContext('2d');
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Find bounds for scaling
        const xValues = projection.map(p => p[0]);
        const yValues = projection.map(p => p[1]);
        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        const yMin = Math.min(...yValues);
        const yMax = Math.max(...yValues);
        
        const scaleX = (canvas.width - 40) / (xMax - xMin || 1);
        const scaleY = (canvas.height - 40) / (yMax - yMin || 1);
        const scale = Math.min(scaleX, scaleY);
        
        // Draw points
        projection.forEach((point, i) => {
            const x = 20 + (point[0] - xMin) * scale;
            const y = canvas.height - 20 - (point[1] - yMin) * scale;
            
            ctx.fillStyle = '#3498db';
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, 2 * Math.PI);
            ctx.fill();
        });
        
        // Add title
        ctx.fillStyle = '#333';
        ctx.font = '14px Arial';
        ctx.fillText('PCA Projection of Item Embeddings', 20, 20);
        ctx.font = '12px Arial';
        ctx.fillText(`Showing ${projection.length} items`, 20, 40);
    }

    async test() {
        if (!this.model) {
            this.updateStatus('Please train a model first!');
            return;
        }

        this.updateStatus('Finding qualified user...');
        
        const qualifiedUser = this.data.getRandomQualifiedUser();
        if (!qualifiedUser) {
            this.updateStatus('No qualified users found (need users with ≥20 ratings)');
            return;
        }
        
        const userIndex = this.data.userMap.get(qualifiedUser);
        this.updateStatus(`Generating recommendations for user ${qualifiedUser}...`);
        
        try {
            // Get user's top rated movies
            const topRated = this.data.getUserTopRated(qualifiedUser);
            
            // Generate recommendations
            const candidateIndices = this.data.getUnratedItemIndices(qualifiedUser);
            const baselineRecs = await this.getRecommendations(
                new BaselineTwoTowerModel(this.model.config), 
                userIndex, 
                candidateIndices
            );
            
            const deepRecs = await this.getRecommendations(
                this.model, 
                userIndex, 
                candidateIndices
            );
            
            this.renderComparisonTable(qualifiedUser, topRated, baselineRecs, deepRecs);
            this.updateStatus(`Recommendations generated for user ${qualifiedUser}`);
            
        } catch (error) {
            this.updateStatus(`Error generating recommendations: ${error.message}`);
        }
    }

    async getRecommendations(model, userIndex, candidateIndices) {
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
                score: score[0]
            })));
            
            tf.dispose([userTensor, itemTensor, batchScores]);
        }
        
        // Return top 10 by score
        return scores.sort((a, b) => b.score - a.score)
                    .slice(0, 10)
                    .map(item => this.data.itemReverseMap.get(item.itemIndex));
    }

    renderComparisonTable(userId, topRated, baselineRecs, deepRecs) {
        const container = document.getElementById('comparisonResults');
        
        let html = `
            <h3>Recommendation Comparison for User ${userId}</h3>
            <table class="comparison-table">
                <tr>
                    <th>Top 10 Rated (Historical)</th>
                    <th>Baseline Recommendations</th>
                    <th>Deep Model Recommendations</th>
                </tr>
        `;
        
        for (let i = 0; i < 10; i++) {
            const ratedItem = topRated[i] ? this.data.getMovie(topRated[i].itemId) : null;
            const baselineItem = baselineRecs[i] ? this.data.getMovie(baselineRecs[i]) : null;
            const deepItem = deepRecs[i] ? this.data.getMovie(deepRecs[i]) : null;
            
            const ratedText = ratedItem ? 
                `${ratedItem.title} (${ratedItem.year})<br><small>Rating: ${topRated[i].rating} • ${ratedItem.genreNames.join(', ')}</small>` : 
                '';
                
            const baselineText = baselineItem ? 
                `${baselineItem.title} (${baselineItem.year})<br><small>${baselineItem.genreNames.join(', ')}</small>` : 
                '';
                
            const deepText = deepItem ? 
                `${deepItem.title} (${deepItem.year})<br><small>${deepItem.genreNames.join(', ')}</small>` : 
                '';
            
            html += `
                <tr>
                    <td>${ratedText}</td>
                    <td>${baselineText}</td>
                    <td>${deepText}</td>
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
