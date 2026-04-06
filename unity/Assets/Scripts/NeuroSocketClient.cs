using System;
using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

/// <summary>
/// WebSocket client care conectează Unity la Python NeuroGame server.
///
/// Flow:
///   1. Se conectează la ws://localhost:8765
///   2. Primește "init" → teleportează agenții la pozițiile de spawn
///   3. Trimite "ready" → primește "step" → lerp agenți la noile poziții
///   4. Când animația se termină → trimite "ready" → repeat
///   5. Pe "done" → urmează "init" pentru episodul următor
///
/// Atașează-l pe un GameObject din scenă (ex: NeuroGameManager).
/// Asignează preyTransform (Umbra) și predTransform (Fenrir) din Inspector.
/// </summary>
public class NeuroSocketClient : MonoBehaviour
{
    [Header("Server")]
    public string serverUrl = "ws://localhost:8765";

    [Header("Agenți — drag din Hierarchy")]
    public Transform preyTransform;   // Umbra (verde)
    public Transform predTransform;   // Fenrir (roșu)

    [Header("Arena")]
    [Tooltip("Dimensiunea arenei în unități Unity (trebuie să coincidă cu pereții din scenă)")]
    public float unityArenaSize = 8f;

    [Header("Animație")]
    [Tooltip("Secunde pentru interpolarea la noua poziție per turn")]
    public float moveTime = 0.12f;

    // ─── WebSocket ────────────────────────────────────────────────
    private ClientWebSocket _ws;
    private CancellationTokenSource _cts;
    private readonly ConcurrentQueue<string> _incoming = new ConcurrentQueue<string>();

    // ─── Stare arenă (primită de la Python) ──────────────────────
    private float _pyArenaW = 30f;
    private float _pyArenaH = 30f;

    // ─── Lerp ─────────────────────────────────────────────────────
    private Vector3 _preyStart, _preyTarget;
    private Vector3 _predStart, _predTarget;
    private float _lerpTimer;
    private bool _isMoving;

    // ─── Stats (opțional — afișat în Console) ────────────────────
    private int _episode;
    private int _turn;

    // ─── Unity lifecycle ──────────────────────────────────────────

    void Start()
    {
        if (preyTransform == null || predTransform == null)
        {
            Debug.LogError("[NeuroSocket] preyTransform sau predTransform nu sunt asignate în Inspector!");
            return;
        }
        _preyTarget = preyTransform.position;
        _predTarget = predTransform.position;

        ConnectAsync().ContinueWith(t =>
        {
            if (t.IsFaulted)
                Debug.LogError($"[NeuroSocket] ConnectAsync failed: {t.Exception?.GetBaseException().Message}");
        });
    }

    void Update()
    {
        // Procesează mesajele primite pe main thread (necesar pentru Transform)
        while (_incoming.TryDequeue(out string raw))
            ProcessMessage(raw);

        // Lerp agenți spre pozițiile țintă
        if (_isMoving)
        {
            _lerpTimer += Time.deltaTime;
            float t = Mathf.Clamp01(_lerpTimer / Mathf.Max(moveTime, 0.001f));

            if (preyTransform) preyTransform.position = Vector3.Lerp(_preyStart, _preyTarget, t);
            if (predTransform) predTransform.position = Vector3.Lerp(_predStart, _predTarget, t);

            if (t >= 1f)
            {
                _isMoving = false;
                // Animația s-a terminat — server-ul poate trimite următorul turn
                SendReadyAsync().ContinueWith(task =>
                {
                    if (task.IsFaulted)
                        Debug.LogWarning($"[NeuroSocket] SendReady failed: {task.Exception?.GetBaseException().Message}");
                });
            }
        }
    }

    void OnDestroy()
    {
        _cts?.Cancel();
        _ws?.Dispose();
    }

    // ─── Conectare ────────────────────────────────────────────────

    async Task ConnectAsync()
    {
        _cts = new CancellationTokenSource();
        _ws = new ClientWebSocket();

        try
        {
            await _ws.ConnectAsync(new Uri(serverUrl), _cts.Token);
            Debug.Log($"[NeuroSocket] Conectat la {serverUrl}");
            _ = ReceiveLoopAsync();
        }
        catch (Exception e)
        {
            Debug.LogError($"[NeuroSocket] Conexiune eșuată: {e.Message}\nPornește server-ul cu: python -m server.neuro_server");
        }
    }

    // ─── Receive loop (background thread) ────────────────────────

    async Task ReceiveLoopAsync()
    {
        var buffer = new byte[8192];
        var sb = new StringBuilder();

        while (_ws != null && _ws.State == WebSocketState.Open)
        {
            try
            {
                sb.Clear();
                WebSocketReceiveResult result;
                do
                {
                    result = await _ws.ReceiveAsync(new ArraySegment<byte>(buffer), _cts.Token);
                    sb.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
                } while (!result.EndOfMessage);

                if (result.MessageType == WebSocketMessageType.Text)
                    _incoming.Enqueue(sb.ToString());
                else if (result.MessageType == WebSocketMessageType.Close)
                    break;
            }
            catch (OperationCanceledException) { break; }
            catch (Exception e)
            {
                Debug.LogWarning($"[NeuroSocket] Receive error: {e.Message}");
                break;
            }
        }
        Debug.Log("[NeuroSocket] Receive loop terminat.");
    }

    // ─── Send "ready" ─────────────────────────────────────────────

    async Task SendReadyAsync()
    {
        if (_ws == null || _ws.State != WebSocketState.Open) return;

        byte[] bytes = Encoding.UTF8.GetBytes("{\"type\":\"ready\"}");
        await _ws.SendAsync(
            new ArraySegment<byte>(bytes),
            WebSocketMessageType.Text,
            endOfMessage: true,
            cancellationToken: _cts.Token
        );
    }

    // ─── Procesare mesaje (main thread) ──────────────────────────

    void ProcessMessage(string raw)
    {
        var msg = JsonUtility.FromJson<NeuroMsg>(raw);
        if (msg == null) return;

        switch (msg.type)
        {
            case "init":
                HandleInit(msg);
                break;
            case "step":
                HandleStep(msg);
                break;
            default:
                Debug.LogWarning($"[NeuroSocket] Tip mesaj necunoscut: {msg.type}");
                break;
        }
    }

    void HandleInit(NeuroMsg msg)
    {
        _pyArenaW = msg.arena_w > 0 ? msg.arena_w : _pyArenaW;
        _pyArenaH = msg.arena_h > 0 ? msg.arena_h : _pyArenaH;
        _episode = msg.episode;
        _isMoving = false;

        // Teleportare la pozițiile de spawn (fără lerp la start episod)
        Vector3 preySpawn = PyToUnity(msg.prey_pos[0], msg.prey_pos[1]);
        Vector3 predSpawn = PyToUnity(msg.pred_pos[0], msg.pred_pos[1]);

        if (preyTransform) preyTransform.position = preySpawn;
        if (predTransform) predTransform.position = predSpawn;

        _preyTarget = preySpawn;
        _predTarget = predSpawn;

        Debug.Log($"[NeuroSocket] Episod {_episode} | Arena {_pyArenaW}×{_pyArenaH}");

        // Pornește primul turn
        SendReadyAsync();
    }

    void HandleStep(NeuroMsg msg)
    {
        _turn = msg.turn;

        _preyStart = preyTransform ? preyTransform.position : Vector3.zero;
        _predStart = predTransform ? predTransform.position : Vector3.zero;

        _preyTarget = PyToUnity(msg.prey_pos[0], msg.prey_pos[1]);
        _predTarget = PyToUnity(msg.pred_pos[0], msg.pred_pos[1]);

        _lerpTimer = 0f;
        _isMoving = true;

        if (msg.done)
            Debug.Log($"[NeuroSocket] Episod {msg.episode} terminat: {msg.info} | turn {msg.turn}");
        // Nota: "ready" se trimite după animație (în Update), care va declanșa "init" pentru episodul următor
    }

    // ─── Coordonate Python → Unity ───────────────────────────────

    Vector3 PyToUnity(float px, float py)
    {
        // Python: origine stânga-jos, (0,0)→(arena_w, arena_h)
        // Unity: centrat la origine, (-size/2, -size/2)→(+size/2, +size/2)
        float ux = (px / _pyArenaW) * unityArenaSize - unityArenaSize * 0.5f;
        float uy = (py / _pyArenaH) * unityArenaSize - unityArenaSize * 0.5f;
        return new Vector3(ux, uy, 0f);
    }
}

// ─── DTO pentru JsonUtility ────────────────────────────────────────

[Serializable]
public class NeuroMsg
{
    public string type;
    public int episode;
    public int turn;
    public float[] prey_pos;
    public float[] pred_pos;
    public float prey_stamina;
    public float prey_reward;
    public float pred_reward;
    public bool done;
    public string info;
    // init only
    public float arena_w;
    public float arena_h;
}
