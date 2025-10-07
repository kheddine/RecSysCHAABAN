/**
 * MovieLens Data Loader and Parser
 * Handles loading and processing of u.data and u.item files
 */
class MovieLensData {
    constructor() {
        this.interactions = [];
        this.items = new Map();
        this.userMap = new Map();
        this.itemMap = new Map();
        this.userReverseMap = new Map();
        this.itemReverseMap = new Map();
        this.userRatedItems = new Map();
        this.userTopRated = new Map();
        this.genresList = [
            'Unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 
            'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
            'Sci-Fi', 'Thriller', 'War', 'Western'
        ];
    }

    /**
     * Load and parse all MovieLens data
     */
    async loadData() {
        await this.loadInteractions();
        await this.loadItems();
        this.buildIndices();
        this.analyzeUserBehavior();
        return this.getStats();
    }

    /**
     * Load and parse u.data interactions file
     */
    async loadInteractions() {
        try {
            const response = await fetch('data/u.data');
            const text = await response.text();
            const lines = text.trim().split('\n');
            
            this.interactions = lines.map(line => {
                const [userId, itemId, rating, timestamp] = line.split('\t');
                return {
                    userId: parseInt(userId),
                    itemId: parseInt(itemId),
                    rating: parseFloat(rating),
                    timestamp: parseInt(timestamp)
                };
            });
            
            console.log(`Loaded ${this.interactions.length} interactions`);
        } catch (error) {
            throw new Error(`Failed to load interactions: ${error.message}`);
        }
    }

    /**
     * Load and parse u.item movie metadata file
     */
    async loadItems() {
        try {
            const response = await fetch('data/u.item');
            const text = await response.text();
            const lines = text.trim().split('\n');
            
            lines.forEach(line => {
                const parts = line.split('|');
                const itemId = parseInt(parts[0]);
                const title = parts[1];
                
                // Extract year from title (format: "Movie Title (YYYY)")
                const yearMatch = title.match(/\((\d{4})\)$/);
                const year = yearMatch ? parseInt(yearMatch[1]) : 0;
                
                // Parse genres (19 binary indicators)
                const genres = parts.slice(5, 24).map(g => parseInt(g));
                
                this.items.set(itemId, {
                    title: title.replace(/\(\d{4}\)$/, '').trim(),
                    year: year,
                    genres: genres,
                    genreNames: this.getGenreNames(genres)
                });
            });
            
            console.log(`Loaded ${this.items.size} items`);
        } catch (error) {
            throw new Error(`Failed to load items: ${error.message}`);
        }
    }

    /**
     * Convert binary genre vector to genre names
     */
    getGenreNames(genres) {
        return genres
            .map((active, index) => active ? this.genresList[index] : null)
            .filter(genre => genre !== null);
    }

    /**
     * Build user and item indices for model training
     */
    buildIndices() {
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
        
        console.log(`Built indices for ${this.userMap.size} users and ${this.itemMap.size} items`);
    }

    /**
     * Analyze user behavior and compute top-rated movies
     */
    analyzeUserBehavior() {
        // Build user -> rated items mapping
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
        
        console.log(`Analyzed behavior for ${this.userRatedItems.size} users`);
    }

    /**
     * Get dataset statistics
     */
    getStats() {
        return {
            interactions: this.interactions.length,
            items: this.items.size,
            users: this.userMap.size,
            qualifiedUsers: this.getQualifiedUsers().length
        };
    }

    /**
     * Get users with at least 20 ratings for testing
     */
    getQualifiedUsers() {
        return Array.from(this.userRatedItems.entries())
            .filter(([_, interactions]) => interactions.length >= 20)
            .map(([userId]) => userId);
    }

    /**
     * Get random qualified user for testing
     */
    getRandomQualifiedUser() {
        const qualifiedUsers = this.getQualifiedUsers();
        if (qualifiedUsers.length === 0) return null;
        return qualifiedUsers[Math.floor(Math.random() * qualifiedUsers.length)];
    }

    /**
     * Get user's top rated movies
     */
    getUserTopRated(userId) {
        return this.userTopRated.get(userId) || [];
    }

    /**
     * Get movie information by ID
     */
    getMovie(itemId) {
        return this.items.get(itemId);
    }

    /**
     * Get genre vector for item ID
     */
    getGenreVector(itemId) {
        const item = this.items.get(itemId);
        return item ? item.genres : new Array(19).fill(0);
    }

    /**
     * Prepare training data with limited interactions
     */
    getTrainingData(maxInteractions = 80000) {
        const trainingInteractions = this.interactions
            .slice(0, maxInteractions)
            .map(interaction => ({
                userId: this.userMap.get(interaction.userId),
                itemId: this.itemMap.get(interaction.itemId),
                rating: interaction.rating
            }));
        
        return trainingInteractions;
    }

    /**
     * Get all item indices for recommendation
     */
    getAllItemIndices() {
        return Array.from({length: this.itemMap.size}, (_, i) => i);
    }

    /**
     * Get item indices that user has not rated
     */
    getUnratedItemIndices(userId, excludeRated = true) {
        const userIndex = this.userMap.get(userId);
        const allIndices = this.getAllItemIndices();
        
        if (!excludeRated) return allIndices;
        
        const ratedItems = this.userRatedItems.get(userId) || [];
        const ratedIndices = new Set(ratedItems.map(interaction => 
            this.itemMap.get(interaction.itemId)
        ));
        
        return allIndices.filter(idx => !ratedIndices.has(idx));
    }
}
