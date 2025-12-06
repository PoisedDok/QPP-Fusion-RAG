package qpp;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;

import java.util.*;

/**
 * Bridge between Python and Java QPP implementations.
 * Accepts JSON input via stdin, executes QPP methods, returns JSON output.
 * 
 * Usage: java -cp target/SupervisedRLM-1.0-SNAPSHOT.jar:target/dependency/* qpp.QPPBridge
 * 
 * Input JSON format:
 * {
 *   "query": "search query text",
 *   "documents": [{"content": "...", "score": 0.95}, ...],
 *   "retriever_name": "BM25",
 *   "methods": ["nqc", "smv", "wig", ...]
 * }
 * 
 * Output JSON format:
 * {
 *   "query": "...",
 *   "retriever_name": "...",
 *   "qpp_scores": {"nqc": 0.75, "smv": 0.82, ...},
 *   "methods_used": ["nqc", "smv", ...],
 *   "processing_time_ms": 123.45
 * }
 */
public class QPPBridge {
    
    private static final Gson gson = new Gson();
    
    public static void main(String[] args) {
        try {
            // Read JSON input from stdin
            Scanner scanner = new Scanner(System.in);
            StringBuilder jsonBuilder = new StringBuilder();
            while (scanner.hasNextLine()) {
                jsonBuilder.append(scanner.nextLine());
            }
            scanner.close();
            
            String inputJson = jsonBuilder.toString();
            JsonObject input = gson.fromJson(inputJson, JsonObject.class);
            
            // Extract parameters
            String query = input.get("query").getAsString();
            JsonArray docsArray = input.getAsJsonArray("documents");
            String retrieverName = input.has("retriever_name") 
                ? input.get("retriever_name").getAsString() 
                : "unknown";
            
            // Extract methods list
            List<String> methods = new ArrayList<>();
            if (input.has("methods")) {
                JsonArray methodsArray = input.getAsJsonArray("methods");
                for (JsonElement elem : methodsArray) {
                    methods.add(elem.getAsString());
                }
            } else {
                // Default to all methods
                methods = Arrays.asList(
                    "nqc", "smv", "wig", "SigmaMax", "SigmaX", "RSD", "UEF",
                    "MaxIDF", "avgidf", "cumnqc", "snqc", "dense-qpp", "dense-qpp-m"
                );
            }
            
            // Parse documents and scores
            List<Float> scores = new ArrayList<>();
            for (JsonElement docElem : docsArray) {
                JsonObject doc = docElem.getAsJsonObject();
                float score = doc.has("score") ? doc.get("score").getAsFloat() : 0.0f;
                scores.add(score);
            }
            
            // Execute QPP methods
            long startTime = System.currentTimeMillis();
            Map<String, Double> qppScores = computeQPPScores(query, scores, methods);
            double processingTimeMs = (System.currentTimeMillis() - startTime);
            
            // Build output JSON
            JsonObject output = new JsonObject();
            output.addProperty("query", query);
            output.addProperty("retriever_name", retrieverName);
            
            JsonObject scoresObj = new JsonObject();
            for (Map.Entry<String, Double> entry : qppScores.entrySet()) {
                scoresObj.addProperty(entry.getKey(), entry.getValue());
                // Also add per-method output
                output.addProperty("score_" + entry.getKey().replace("-", "_"), entry.getValue());
            }
            output.add("qpp_scores", scoresObj);
            
            JsonArray methodsUsed = new JsonArray();
            for (String method : qppScores.keySet()) {
                methodsUsed.add(method);
            }
            output.add("methods_used", methodsUsed);
            output.addProperty("processing_time_ms", processingTimeMs);
            
            // Add predictions
            JsonObject predictions = aggregatePredictions(qppScores);
            output.add("predictions", predictions);
            
            // Output JSON to stdout
            System.out.println(gson.toJson(output));
            
        } catch (Exception e) {
            // Error output
            JsonObject error = new JsonObject();
            error.addProperty("error", e.getMessage());
            error.addProperty("error_type", e.getClass().getSimpleName());
            System.err.println(gson.toJson(error));
            System.exit(1);
        }
    }
    
    private static Map<String, Double> computeQPPScores(
        String query, 
        List<Float> scores, 
        List<String> methods
    ) {
        Map<String, Double> qppScores = new LinkedHashMap<>();
        
        // CRITICAL FIX: Pre-normalize scores to [0,1] range per query
        // This is what research papers do - QPP methods assume normalized scores
        List<Float> normalizedScores = normalizeScoresToUnitRange(scores);
        
        for (String method : methods) {
            try {
                // RSD and IDF methods use raw scores (scale-invariant or query-based)
                // All other methods use normalized scores
                double score;
                if (method.equalsIgnoreCase("rsd") || 
                    method.equalsIgnoreCase("maxidf") || 
                    method.equalsIgnoreCase("avgidf")) {
                    score = computeSingleMethod(method, query, scores);
                } else {
                    score = computeSingleMethod(method, query, normalizedScores);
                }
                qppScores.put(method, score);
            } catch (Exception e) {
                System.err.println("Warning: Method " + method + " failed: " + e.getMessage());
                qppScores.put(method, 0.0);
            }
        }
        
        return qppScores;
    }
    
    /**
     * Normalize scores to [0,1] range using min-max normalization.
     * This is critical for QPP methods to work across different retriever score ranges.
     */
    private static List<Float> normalizeScoresToUnitRange(List<Float> scores) {
        if (scores.isEmpty()) return scores;
        
        float minScore = Float.MAX_VALUE;
        float maxScore = Float.MIN_VALUE;
        for (float s : scores) {
            minScore = Math.min(minScore, s);
            maxScore = Math.max(maxScore, s);
        }
        
        float range = maxScore - minScore;
        if (range < 1e-10) {
            // All scores are the same - return uniform 0.5
            List<Float> uniform = new ArrayList<>();
            for (int i = 0; i < scores.size(); i++) uniform.add(0.5f);
            return uniform;
        }
        
        List<Float> normalized = new ArrayList<>();
        for (float s : scores) {
            normalized.add((s - minScore) / range);
        }
        return normalized;
    }
    
    private static double computeSingleMethod(String method, String query, List<Float> scores) {
        // Normalize method name for case-insensitive matching
        String m = method.toLowerCase();
        
        switch (m) {
            case "nqc":
                return computeNQC(scores);
            case "smv":
                return computeSMV(scores);
            case "wig":
                return computeWIG(query, scores);
            case "sigmamax":
            case "sigma_max":
                return computeSigmaMax(query, scores);
            case "sigmax":
            case "sigma_x":
            case "sigma(%)":
                return computeSigmaX(scores);
            case "rsd":
                return computeRSD(scores);
            case "uef":
                return computeUEF(scores);
            case "maxidf":
                return computeMaxIDF(query);
            case "avgidf":
                return computeAvgIDF(query);
            case "cumnqc":
                return computeCumNQC(scores);
            case "snqc":
            case "scnqc":
                return computeSNQC(scores);
            case "dense-qpp":
            case "denseqpp":
                return computeDenseQPP(scores);
            case "dense-qpp-m":
            case "denseqppm":
                return computeDenseQPPMatryoshka(scores);
            default:
                throw new IllegalArgumentException("Unknown QPP method: " + method);
        }
    }
    
    // ========================================================================
    // QPP Method Implementations (Simplified - mirrors Java implementations)
    // ========================================================================
    
    private static double computeNQC(List<Float> scores) {
        // NQC: Normalized Query Commitment
        // With pre-normalized [0,1] scores, variance is naturally bounded [0, 0.25]
        if (scores.size() < 2) return 0.5;
        
        int k = Math.min(50, scores.size());
        List<Float> topK = getTopK(scores, k);
        
        double std = stdDev(topK);
        
        // High std = spread scores = hard query = low NQC
        // Low std = tight cluster = easy query = high NQC
        // With [0,1] scores, std is typically 0.0-0.35
        return 1.0 - Math.min(1.0, std * 3);
    }
    
    private static double computeSMV(List<Float> scores) {
        // SMV: Similarity Mean Variance
        // Measures score dispersion relative to mean
        if (scores.size() < 2) return 0.5;
        
        int k = Math.min(10, scores.size());
        List<Float> topK = getTopK(scores, k);
        
        double muHat = mean(topK);
        if (muHat < 0.01) return 0.0;
        
        // Coefficient of variation (std/mean) - scale-invariant dispersion measure
        double cv = stdDev(topK) / muHat;
        
        // Low CV = consistent high scores = easy query = high SMV
        // High CV = inconsistent = hard query = low SMV
        return 1.0 - Math.min(1.0, cv);
    }
    
    private static double computeWIG(String query, List<Float> scores) {
        // WIG: Weighted Information Gain
        // With pre-normalized [0,1] scores, measures how much top-k exceeds corpus average
        if (scores.isEmpty()) return 0.0;
        
        int k = Math.min(20, scores.size());
        List<Float> topK = getTopK(scores, k);
        
        // Corpus baseline = mean of all scores (now in [0,1])
        double corpusBaseline = mean(scores);
        double topKMean = mean(topK);
        
        // WIG = (top-k mean - corpus mean) / query_complexity
        String[] terms = query.toLowerCase().split("\\s+");
        int numTerms = Math.max(1, terms.length);
        
        // With normalized scores, WIG is naturally bounded
        // Positive = top results much better than average
        // Negative = top results not much better (hard query)
        double wig = (topKMean - corpusBaseline) / Math.sqrt(numTerms);
        
        // Transform to [0,1] - WIG typically ranges from -0.5 to 0.5 with normalized scores
        return 0.5 + wig;
    }
    
    private static double computeSigmaMax(String query, List<Float> scores) {
        // SigmaMax: Maximum std across growing windows
        // With pre-normalized [0,1] scores, std is bounded
        if (scores.size() < 2) return 0.5;
        
        int k = Math.min(50, scores.size());
        List<Float> topK = getTopK(scores, k);
        
        double maxStd = 0.0;
        for (int i = 2; i <= topK.size(); i++) {
            List<Float> window = topK.subList(0, i);
            maxStd = Math.max(maxStd, stdDev(window));
        }
        
        // With [0,1] scores, maxStd is typically 0.0-0.4
        // Low maxStd = consistent results = easy query
        return 1.0 - Math.min(1.0, maxStd * 2.5);
    }
    
    private static double computeSigmaX(List<Float> scores) {
        // SigmaX: Std of scores above threshold
        // With pre-normalized [0,1] scores, threshold is meaningful
        if (scores.size() < 2) return 0.5;
        
        int k = Math.min(50, scores.size());
        List<Float> topK = getTopK(scores, k);
        
        // Filter scores >= 50% of top score
        float topScore = topK.get(0);
        float threshold = topScore * 0.5f;
        
        List<Float> filtered = new ArrayList<>();
        for (float s : topK) {
            if (s >= threshold) filtered.add(s);
        }
        
        if (filtered.size() < 2) return 0.5;
        
        double std = stdDev(filtered);
        // With [0,1] scores, std of filtered is typically 0.0-0.25
        return 1.0 - Math.min(1.0, std * 4);
    }
    
    private static double computeRSD(List<Float> scores) {
        if (scores.size() < 2) return 0.5;
        
        double mean = mean(scores);
        double std = stdDev(scores);
        
        if (std == 0) return Math.min(1.0, mean);
        
        double skewness = 0.0;
        for (float s : scores) {
            skewness += Math.pow((s - mean), 3);
        }
        skewness /= (scores.size() * Math.pow(std, 3));
        
        return 1.0 / (1.0 + Math.exp(-skewness));
    }
    
    private static double computeUEF(List<Float> scores) {
        // UEF: Utility Estimation Framework
        // With pre-normalized [0,1] scores, this now works properly
        if (scores.isEmpty()) return 0.0;
        
        List<Float> sorted = getTopK(scores, scores.size());
        int k = Math.min(20, sorted.size());
        
        double utility = 0.0;
        double weightSum = 0.0;
        for (int i = 0; i < k; i++) {
            double weight = 1.0 / (i + 1);  // DCG-style weighting
            utility += sorted.get(i) * weight;
            weightSum += weight;
        }
        
        // With normalized [0,1] scores, ratio is naturally in [0,1]
        return utility / weightSum;
    }
    
    private static double computeMaxIDF(String query) {
        // Split on whitespace and get unique terms
        String[] terms = query.toLowerCase().split("\\s+");
        java.util.Set<String> uniqueTerms = new java.util.HashSet<>();
        int maxLen = 0;
        for (String t : terms) {
            uniqueTerms.add(t);
            maxLen = Math.max(maxLen, t.length());
        }
        // Proxy for IDF: term diversity + max term length
        double termDiversity = (double) uniqueTerms.size() / terms.length;
        double maxIDF = Math.log(1 + uniqueTerms.size()) * (1 + termDiversity) + Math.log(1 + maxLen) * 0.5;
        return 1.0 / (1.0 + Math.exp(-maxIDF + 2));
    }
    
    private static double computeAvgIDF(String query) {
        // Split on whitespace and analyze term characteristics
        String[] terms = query.toLowerCase().split("\\s+");
        java.util.Set<String> uniqueTerms = new java.util.HashSet<>();
        double totalLen = 0;
        for (String t : terms) {
            uniqueTerms.add(t);
            totalLen += t.length();
        }
        // Proxy for avg IDF: avg term length + uniqueness ratio
        double avgLen = totalLen / terms.length;
        double termDiversity = (double) uniqueTerms.size() / terms.length;
        double avgIDF = Math.log(1 + avgLen) * termDiversity + Math.log(1 + uniqueTerms.size()) * 0.5;
        return 1.0 / (1.0 + Math.exp(-avgIDF + 1.5));
    }
    
    private static double computeCumNQC(List<Float> scores) {
        // CumNQC: Cumulative NQC - average std across growing windows
        // With pre-normalized [0,1] scores, this is naturally bounded
        if (scores.size() < 2) return 0.5;
        
        int kMax = Math.min(50, scores.size());
        List<Float> topK = getTopK(scores, kMax);
        
        double cumStd = 0.0;
        int count = 0;
        for (int k = 2; k <= kMax; k++) {
            List<Float> window = topK.subList(0, k);
            cumStd += stdDev(window);
            count++;
        }
        
        double avgStd = cumStd / Math.max(1, count);
        // With [0,1] scores, avgStd is typically 0.0-0.3
        return 1.0 - Math.min(1.0, avgStd * 3);
    }
    
    private static double computeSNQC(List<Float> scores) {
        // SNQC: Calibrated NQC using coefficient of variation
        // With pre-normalized [0,1] scores, simplified formula works
        if (scores.size() < 2) return 0.5;
        
        int k = Math.min(50, scores.size());
        List<Float> topK = getTopK(scores, k);
        
        double mean = mean(topK);
        double std = stdDev(topK);
        
        if (mean < 0.01) return 0.0;
        
        // Coefficient of variation scaled by mean level
        double cv = std / mean;
        double snqc = mean * (1.0 - cv);
        
        // With [0,1] scores, snqc is typically 0.0-0.8
        return Math.max(0, Math.min(1.0, snqc * 1.2));
    }
    
    private static double computeDenseQPP(List<Float> scores) {
        // DenseQPP: Measures clustering tightness of top results
        // With pre-normalized [0,1] scores, variance is naturally bounded
        if (scores.size() < 2) return 0.5;
        
        int k = Math.min(5, scores.size());
        List<Float> topK = getTopK(scores, k);
        
        // Std dev of top-k (in [0,1] range, so std is typically 0-0.3)
        double std = stdDev(topK);
        
        // Low std = tight clustering = easy query = high QPP
        // High std = spread out = hard query = low QPP
        // With normalized scores, std is typically 0.0-0.3
        return 1.0 - Math.min(1.0, std * 3);
    }
    
    private static double computeDenseQPPMatryoshka(List<Float> scores) {
        // DenseQPP-M: Multi-scale analysis of clustering consistency
        // With pre-normalized [0,1] scores, this works properly
        if (scores.size() < 2) return 0.5;
        
        int kMax = Math.min(5, scores.size());
        List<Float> topK = getTopK(scores, kMax);
        
        // Compute std at different scales
        double[] scaleStd = new double[kMax];
        for (int i = 1; i <= kMax; i++) {
            List<Float> window = topK.subList(0, i);
            scaleStd[i-1] = (i > 1) ? stdDev(window) : 0.0;
        }
        
        // Measure consistency: how stable is std across scales?
        double meanStd = 0.0;
        for (double s : scaleStd) meanStd += s;
        meanStd /= kMax;
        
        double stdVariance = 0.0;
        for (double s : scaleStd) stdVariance += Math.pow(s - meanStd, 2);
        stdVariance /= kMax;
        
        // Low variance in std across scales = consistent clustering = easy query
        double consistency = 1.0 - Math.min(1.0, Math.sqrt(stdVariance) * 5);
        
        // Combine with overall tightness
        double tightness = 1.0 - Math.min(1.0, meanStd * 3);
        
        return 0.4 * consistency + 0.6 * tightness;
    }
    
    // ========================================================================
    // Utility Methods
    // ========================================================================
    
    private static List<Float> getTopK(List<Float> scores, int k) {
        List<Float> sorted = new ArrayList<>(scores);
        sorted.sort(Collections.reverseOrder());
        return sorted.subList(0, Math.min(k, sorted.size()));
    }
    
    private static double mean(List<Float> values) {
        if (values.isEmpty()) return 0.0;
        double sum = 0.0;
        for (float v : values) sum += v;
        return sum / values.size();
    }
    
    private static double variance(List<Float> values) {
        if (values.size() < 2) return 0.0;
        double mean = mean(values);
        double sumSq = 0.0;
        for (float v : values) {
            sumSq += Math.pow(v - mean, 2);
        }
        return sumSq / values.size();
    }
    
    private static double stdDev(List<Float> values) {
        return Math.sqrt(variance(values));
    }
    
    private static JsonObject aggregatePredictions(Map<String, Double> qppScores) {
        if (qppScores.isEmpty()) {
            JsonObject pred = new JsonObject();
            pred.addProperty("difficulty_estimate", 1.0);
            pred.addProperty("retrieval_quality", 0.0);
            pred.addProperty("recommended_action", "fallback");
            pred.addProperty("confidence", 0.0);
            return pred;
        }
        
        double meanQPP = qppScores.values().stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0.0);
        
        double variance = 0.0;
        for (double score : qppScores.values()) {
            variance += Math.pow(score - meanQPP, 2);
        }
        variance /= qppScores.size();
        
        double confidence = 1.0 / (1.0 + variance);
        double retrievalQuality = meanQPP;
        double difficultyEstimate = 1.0 - retrievalQuality;
        
        String action;
        if (retrievalQuality >= 0.7) {
            action = "proceed";
        } else if (retrievalQuality >= 0.4) {
            action = "augment";
        } else {
            action = "fallback";
        }
        
        JsonObject pred = new JsonObject();
        pred.addProperty("difficulty_estimate", difficultyEstimate);
        pred.addProperty("retrieval_quality", retrievalQuality);
        pred.addProperty("recommended_action", action);
        pred.addProperty("confidence", confidence);
        
        return pred;
    }
}

