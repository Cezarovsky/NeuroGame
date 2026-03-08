using UnityEngine;
using System;
using System.Net.Sockets;
using System.Text;
using System.Collections.Generic;

/// <summary>
/// Neuromorphic Event Bus — bridge Unity ↔ Mistral.
///
/// NU trimite date brute de fizică. Trimite spike-uri semantice comprimate:
/// "Ce s-a întâmplat strategic în ultimele N episoade?"
///
/// Mistral primește → răspunde cu attention weights pentru fiecare agent.
/// Agentul nu știe de Mistral — el vede doar că anumite direcții
/// au prioritate mai mare în looming scan (AttentionWeights[]).
/// </summary>
public class NeuromorphicEventBus : MonoBehaviour
{
    public static NeuromorphicEventBus Instance { get; private set; }

    [Header("Conexiune Python bridge")]
    public string host = "127.0.0.1";
    public int port = 5757;

    [Header("Frecvență trimitere la Mistral")]
    [Tooltip("Trimite după fiecare N episoade")]
    public int sendEveryNEpisodes = 10;

    [Header("Referințe agenți (setate automat de Manager)")]
    public NeuroAgent preyAgent;
    public NeuroAgent predatorAgent;

    private List<EpisodeSummary> _buffer = new();
    private TcpClient _client;
    private bool _connected = false;

    void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
    }

    void Start() => TryConnect();

    void TryConnect()
    {
        try
        {
            _client = new TcpClient(host, port);
            _connected = true;
            Debug.Log($"[EventBus] Conectat la Mistral bridge pe {host}:{port}");
        }
        catch (Exception e)
        {
            // Python bridge nu e pornit — funcționăm fără Mistral (Level 0 + 1 only)
            _connected = false;
            Debug.LogWarning($"[EventBus] Bridge offline — rulăm fără Mistral. ({e.Message})");
        }
    }

    public void OnEpisodeStart(int episode) { /* rezervat pentru viitor */ }

    public void OnEpisodeEnd(EpisodeSummary summary)
    {
        _buffer.Add(summary);

        if (_buffer.Count >= sendEveryNEpisodes)
        {
            SendToMistral();
            _buffer.Clear();
        }
    }

    void SendToMistral()
    {
        if (!_connected) return;

        // Comprimăm buffer-ul în spike semantic
        float avgCapture = 0f, avgDuration = 0f;
        float avgPreyLoom = 0f, avgPredLoom = 0f;
        string trend = "unknown";

        foreach (var s in _buffer)
        {
            avgCapture += s.captured ? 1f : 0f;
            avgDuration += s.duration;
            avgPreyLoom += s.preyLoomingPeak;
            avgPredLoom += s.predatorLoomingPeak;
        }
        int n = _buffer.Count;
        avgCapture /= n;
        avgDuration /= n;
        avgPreyLoom /= n;
        avgPredLoom /= n;

        // Trend: lupul devine mai bun sau mai slab?
        float firstHalfCapture = 0f, secondHalfCapture = 0f;
        for (int i = 0; i < n / 2; i++) firstHalfCapture += _buffer[i].captured ? 1f : 0f;
        for (int i = n / 2; i < n; i++) secondHalfCapture += _buffer[i].captured ? 1f : 0f;
        firstHalfCapture /= (n / 2);
        secondHalfCapture /= (n - n / 2);
        trend = secondHalfCapture > firstHalfCapture ? "predator_improving" : "prey_improving";

        // JSON spike semantic
        string spike = JsonUtility.ToJson(new MistralSpikePayload
        {
            episode_range = $"{_buffer[0].episode}-{_buffer[n - 1].episode}",
            capture_rate = avgCapture,
            avg_episode_duration = avgDuration,
            prey_looming_peak_avg = avgPreyLoom,
            predator_looming_peak_avg = avgPredLoom,
            trend = trend,
            request = "update_attention_weights"
        });

        try
        {
            byte[] data = Encoding.UTF8.GetBytes(spike + "\n");
            _client.GetStream().Write(data, 0, data.Length);

            // Citim răspunsul Mistral (attention weights update)
            byte[] response = new byte[4096];
            int bytesRead = _client.GetStream().Read(response, 0, response.Length);
            string json = Encoding.UTF8.GetString(response, 0, bytesRead);

            ApplyAttentionUpdate(JsonUtility.FromJson<MistralAttentionUpdate>(json));
            Debug.Log($"[EventBus] Spike trimis. Trend: {trend}. Mistral a răspuns.");
        }
        catch (Exception e)
        {
            _connected = false;
            Debug.LogWarning($"[EventBus] Conexiune pierdută: {e.Message}");
        }
    }

    void ApplyAttentionUpdate(MistralAttentionUpdate update)
    {
        // Mistral recalibrează ATENȚIA — nu threshold-ul kantian (acela e fix)
        // "Scanează mai mult NW" = weight[NW] = 1.8 în loc de 1.0
        if (preyAgent != null && update.prey_attention != null
            && update.prey_attention.Length == 8)
            preyAgent.AttentionWeights = update.prey_attention;

        if (predatorAgent != null && update.predator_attention != null
            && update.predator_attention.Length == 8)
            predatorAgent.AttentionWeights = update.predator_attention;
    }

    void OnDestroy()
    {
        _client?.Close();
    }
}

// ─── Data structs pentru serializare JSON ───────────────────────────

[Serializable]
public class MistralSpikePayload
{
    public string episode_range;
    public float capture_rate;
    public float avg_episode_duration;
    public float prey_looming_peak_avg;
    public float predator_looming_peak_avg;
    public string trend;
    public string request;
}

[Serializable]
public class MistralAttentionUpdate
{
    public float[] prey_attention;         // 8 direcții: N,NE,E,SE,S,SW,W,NW
    public float[] predator_attention;
    public string reasoning;               // ce a "gândit" Mistral (pentru log)
}
