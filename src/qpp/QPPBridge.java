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
        if (scores.size() < 2) return 0.5;
        
        int k = Math.min(50, scores.size());
        List<Float> topK = getTopK(scores, k);
        
        double mean = mean(topK);
        double variance = variance(topK);
        
        // Normalize
        return 1.0 / (1.0 + Math.exp(-variance));
    }
    
    private static double computeSMV(List<Float> scores) {
        if (scores.size() < 2) return 0.5;
        
        int k = Math.min(10, scores.size());
        List<Float> topK = getTopK(scores, k);
        
        double muHat = mean(topK);
        if (muHat == 0) return 0.0;
        
        double smv = 0.0;
        for (float score : topK) {
            if (score > 0) {
                smv += score * Math.abs(Math.log(score / muHat));
            }
        }
        smv /= topK.size();
        
        return 1.0 / (1.0 + Math.exp(-smv));
    }
    
    private static double computeWIG(String query, List<Float> scores) {
        if (scores.isEmpty()) return 0.0;
        
        int k = Math.min(20, scores.size());
        List<Float> topK = getTopK(scores, k);
        
        // WIG: Weighted Information Gain - measures score deviation from corpus average
        // Without real IDF, use mean of ALL scores as corpus baseline (not just top-k)
        double corpusBaseline = mean(scores);  // Use full score list
        double topKMean = mean(topK);
        
        // WIG is the gain of top-k over corpus baseline, normalized by query complexity
        String[] terms = query.toLowerCase().split("\\s+");
        int numTerms = terms.length;
        java.util.Set<String> uniqueTerms = new java.util.HashSet<>();
        for (String t : terms) uniqueTerms.add(t);
        
        // Incorporate term diversity as query difficulty proxy
        double termDiversity = (double) uniqueTerms.size() / numTerms;
        double wig = (topKMean - corpusBaseline) / (numTerms * termDiversity + 0.1);
        
        return 1.0 / (1.0 + Math.exp(-wig * 10));
    }
    
    private static double computeSigmaMax(String query, List<Float> scores) {
        if (scores.size() < 2) return 0.5;
        
        int k = Math.min(50, scores.size());
        List<Float> topK = getTopK(scores, k);
        
        double maxStd = 0.0;
        for (int i = 2; i <= topK.size(); i++) {
            List<Float> window = topK.subList(0, i);
            double std = stdDev(window);
            maxStd = Math.max(maxStd, std);
        }
        
        int numTerms = query.split("\\s+").length;
        double norm = Math.sqrt(Math.max(1, numTerms));
        
        double sigmaMax = maxStd / norm;
        return 1.0 / (1.0 + Math.exp(-sigmaMax * 2));
    }
    
    private static double computeSigmaX(List<Float> scores) {
        if (scores.size() < 2) return 0.5;
        
        double x = 0.5; // Threshold
        int k = Math.min(50, scores.size());
        List<Float> topK = getTopK(scores, k);
        
        float topScore = topK.get(0);
        List<Float> filtered = new ArrayList<>();
        for (float s : topK) {
            if (s >= topScore * x) {
                filtered.add(s);
            }
        }
        
        if (filtered.isEmpty()) return 0.0;
        
        double std = stdDev(filtered);
        return 1.0 / (1.0 + Math.exp(-std * 2));
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
        if (scores.size() < 2) return 0.5;
        
        int kMax = Math.min(50, scores.size());
        List<Float> topK = getTopK(scores, kMax);
        
        double cumSum = 0.0;
        for (int k = 1; k < kMax; k++) {
            List<Float> window = topK.subList(0, k);
            if (window.size() > 1) {
                cumSum += variance(window);
            }
        }
        
        double cumNQC = cumSum / kMax;
        return 1.0 / (1.0 + Math.exp(-cumNQC * 10));
    }
    
    private static double computeSNQC(List<Float> scores) {
        if (scores.size() < 2) return 0.5;
        
        // Calibrated NQC with alpha, beta, gamma
        double alpha = 0.33, beta = 0.33, gamma = 0.33;
        
        int k = Math.min(50, scores.size());
        List<Float> topK = getTopK(scores, k);
        
        double mean = mean(topK);
        double avgIDF = 1.0; // Simplified
        
        double snqc = 0.0;
        for (float rsv : topK) {
            if (rsv > 0) {
                double factor1 = Math.pow(avgIDF, alpha);
                double factor2 = Math.pow(Math.pow((rsv - mean), 2) / rsv, beta);
                double prod = Math.pow(factor1 * factor2, gamma);
                snqc += prod;
            }
        }
        
        snqc /= topK.size();
        snqc *= avgIDF;
        
        return 1.0 / (1.0 + Math.exp(-snqc * 10));
    }
    
    private static double computeDenseQPP(List<Float> scores) {
        if (scores.size() < 2) return 0.5;
        
        int k = Math.min(5, scores.size());
        List<Float> topK = getTopK(scores, k);
        
        double variance = variance(topK);
        double diameter = Math.sqrt(variance) + 0.01;
        
        // Use sigmoid to preserve variance
        double denseQPP = Math.log(1 + 1 / diameter);
        return 1.0 / (1.0 + Math.exp(-denseQPP + 2));
    }
    
    private static double computeDenseQPPMatryoshka(List<Float> scores) {
        if (scores.size() < 2) return 0.5;
        
        int kMax = Math.min(5, scores.size());
        List<Float> topK = getTopK(scores, kMax);
        
        // Matryoshka: Multi-scale analysis at different k values
        // Measures consistency of clustering across scales
        double[] scaleValues = new double[kMax];
        for (int i = 1; i <= kMax; i++) {
            List<Float> window = topK.subList(0, i);
            double var = variance(window);
            double std = Math.sqrt(var);
            double mean = mean(window);
            // Coefficient of variation at each scale
            scaleValues[i-1] = (mean > 0.001) ? std / mean : std;
        }
        
        // Compute variance of scale values (how consistent is clustering?)
        double scaleVariance = 0.0;
        double scaleMean = 0.0;
        for (double v : scaleValues) scaleMean += v;
        scaleMean /= kMax;
        for (double v : scaleValues) scaleVariance += Math.pow(v - scaleMean, 2);
        scaleVariance /= kMax;
        
        // Low scale variance = consistent clustering = easy query
        // High scale variance = inconsistent = harder query
        double consistency = 1.0 / (1.0 + Math.sqrt(scaleVariance) * 10);
        
        // Combine with average scale (tightness)
        double tightness = 1.0 / (1.0 + scaleMean * 5);
        
        return 0.5 * consistency + 0.5 * tightness;
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

