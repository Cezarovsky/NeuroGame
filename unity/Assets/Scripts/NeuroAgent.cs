using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Agent neuromorphic unificat. Rolul determină totul:
/// - PREY: looming mare = pericol → fuge
/// - PREDATOR: looming mic pe target = oportunitate → atacă
///
/// Același creier (KantianSensor + Q-learning), semantică opusă.
/// Exact cum lupul și iepurele au același sistem nervos dar scopuri inverse.
/// </summary>
public enum AgentRole { Prey, Predator }

public class NeuroAgent : MonoBehaviour
{
    [Header("Identitate")]
    public AgentRole role = AgentRole.Prey;
    public string agentId = "agent_01";

    [Header("Q-Learning")]
    public float learningRate = 0.1f;
    public float discountFactor = 0.95f;
    public float epsilon = 1.0f;
    public float epsilonDecay = 0.9995f;
    public float epsilonMin = 0.05f;
    public float moveSpeed = 3f;

    [Header("Neuromorphic — prag kantian")]
    [Tooltip("Prey: loom > prag = fugi. Predator: loom_on_prey > prag = ataca.")]
    public float loomingThreshold = 0.08f;
    public float reflexSpeed = 6f;
    public float reflexDuration = 0.3f;

    [Header("Grid discretizare")]
    public int gridSize = 10;

    // ─── State intern ───────────────────────────────────────────────
    private const int NUM_ACTIONS = 5;  // 0=stay, 1=up, 2=down, 3=left, 4=right
    private Dictionary<string, float[]> _qTable = new();

    private Rigidbody2D _rb;
    private NeuroAgent _opponent;           // referință la celălalt agent

    private bool _reflexActive = false;
    private float _reflexTimer = 0f;
    private Vector2 _reflexDir = Vector2.zero;

    private int _currentAction = 0;
    private string _currentStateKey = "";

    // ─── Stats publice ───────────────────────────────────────────────
    public float EpisodeReward { get; private set; }
    public int Steps { get; private set; }
    public float Epsilon => epsilon;
    public int QTableSize => _qTable.Count;
    public float CurrentLoomingIndex { get; private set; }
    public bool ReflexActive => _reflexActive;

    // Atenție direcțională — calibrată de Mistral
    // Mistral poate spune "scanează mai mult NW" → attention weight > 1.0 pe acea direcție
    [HideInInspector] public float[] AttentionWeights = new float[8];  // N,NE,E,SE,S,SW,W,NW

    void Awake()
    {
        _rb = GetComponent<Rigidbody2D>();
        for (int i = 0; i < 8; i++) AttentionWeights[i] = 1.0f;
    }

    public void SetOpponent(NeuroAgent opponent) => _opponent = opponent;

    void Update()
    {
        ComputeLoomingAndReflex();
    }

    void FixedUpdate()
    {
        // Level 0: reflex kantian — PRIORITATE ABSOLUTĂ
        if (_reflexActive)
        {
            _rb.linearVelocity = _reflexDir * reflexSpeed;
            return;
        }

        // Level 1: Q-learning
        string stateKey = GetStateKey();
        _currentAction = SelectAction(stateKey);
        _rb.linearVelocity = ActionToVelocity(_currentAction);
        _currentStateKey = stateKey;
    }

    // ─── Neuromorphic: calcul looming ───────────────────────────────

    void ComputeLoomingAndReflex()
    {
        if (_reflexActive)
        {
            _reflexTimer -= Time.deltaTime;
            if (_reflexTimer <= 0f) _reflexActive = false;
            return;
        }

        if (_opponent == null) return;

        float loom = ComputeLoomingOn(_opponent.transform.position,
                                      _opponent.GetComponent<Rigidbody2D>()?.linearVelocity ?? Vector2.zero);
        CurrentLoomingIndex = loom;

        if (role == AgentRole.Prey)
        {
            // Iepurele: looming mare de la lup = reflex de fugă
            if (loom > loomingThreshold)
                TriggerReflex(Flee());
        }
        else
        {
            // Lupul: looming MIC pe iepure (iepurele stă/încetinește) = oportunitate
            // Lupul NU are reflex de fugă — are reflex de ATAC
            // "Looming invers": când iepurele e aproape și lent → spike de atac
            float inverseSignal = (loom < 0.02f && DistanceTo(_opponent) < 4f) ? 1f : 0f;
            if (inverseSignal > 0f)
                TriggerReflex(Attack());
        }
    }

    float ComputeLoomingOn(Vector2 targetPos, Vector2 targetVel)
    {
        Vector2 toTarget = targetPos - (Vector2)transform.position;
        float dist = toTarget.magnitude;
        if (dist < 0.01f) return 999f;

        // Componenta vitezei relative care se apropie
        Vector2 relVel = targetVel - _rb.linearVelocity;
        float approachSpeed = Vector2.Dot(relVel, -toTarget.normalized);

        // Aplicăm attention weights din Mistral (direcția din care vine)
        float angle = Mathf.Atan2(toTarget.y, toTarget.x) * Mathf.Rad2Deg;
        int dirBucket = Mathf.RoundToInt((angle + 360f) / 45f) % 8;
        float attentionMult = AttentionWeights[dirBucket];

        return (approachSpeed / dist) * attentionMult;
    }

    void TriggerReflex(Vector2 dir)
    {
        _reflexActive = true;
        _reflexTimer = reflexDuration;
        _reflexDir = dir;
    }

    Vector2 Flee()
    {
        // Iepurele fuge în direcția opusă lupului
        if (_opponent == null) return Vector2.zero;
        return ((Vector2)transform.position - (Vector2)_opponent.transform.position).normalized;
    }

    Vector2 Attack()
    {
        // Lupul se aruncă spre iepure
        if (_opponent == null) return Vector2.zero;
        return ((Vector2)_opponent.transform.position - (Vector2)transform.position).normalized;
    }

    float DistanceTo(NeuroAgent other) =>
        Vector2.Distance(transform.position, other.transform.position);

    // ─── Q-Learning ─────────────────────────────────────────────────

    string GetStateKey()
    {
        Camera cam = Camera.main;
        Vector3 vp = cam.WorldToViewportPoint(transform.position);
        int gx = Mathf.Clamp(Mathf.FloorToInt(vp.x * gridSize), 0, gridSize - 1);
        int gy = Mathf.Clamp(Mathf.FloorToInt(vp.y * gridSize), 0, gridSize - 1);

        // Direcția spre/dinspre oponent + distanța (bucketizată)
        int distBucket = _opponent == null ? 0 :
            DistanceTo(_opponent) < 2f ? 0 :
            DistanceTo(_opponent) < 5f ? 1 : 2;

        float loom = CurrentLoomingIndex;
        int loomBucket = loom < 0.02f ? 0 : loom < 0.05f ? 1 : loom < 0.1f ? 2 : 3;

        return $"{gx},{gy},{distBucket},{loomBucket}";
    }

    int SelectAction(string stateKey)
    {
        if (!_qTable.ContainsKey(stateKey))
            _qTable[stateKey] = new float[NUM_ACTIONS];

        if (Random.value < epsilon)
            return Random.Range(0, NUM_ACTIONS);

        float[] q = _qTable[stateKey];
        int best = 0;
        for (int i = 1; i < NUM_ACTIONS; i++)
            if (q[i] > q[best]) best = i;
        return best;
    }

    Vector2 ActionToVelocity(int action) => action switch
    {
        1 => Vector2.up * moveSpeed,
        2 => Vector2.down * moveSpeed,
        3 => Vector2.left * moveSpeed,
        4 => Vector2.right * moveSpeed,
        _ => Vector2.zero
    };

    public void ReceiveReward(float reward, string nextStateKey)
    {
        EpisodeReward += reward;
        Steps++;

        if (!_qTable.ContainsKey(_currentStateKey))
            _qTable[_currentStateKey] = new float[NUM_ACTIONS];
        if (!_qTable.ContainsKey(nextStateKey))
            _qTable[nextStateKey] = new float[NUM_ACTIONS];

        float[] q = _qTable[_currentStateKey];
        float[] qNext = _qTable[nextStateKey];
        float maxNext = qNext[0];
        foreach (var v in qNext) if (v > maxNext) maxNext = v;

        q[_currentAction] += learningRate * (reward + discountFactor * maxNext - q[_currentAction]);
        epsilon = Mathf.Max(epsilonMin, epsilon * epsilonDecay);
    }

    public void ResetEpisode()
    {
        EpisodeReward = 0f;
        Steps = 0;
        _reflexActive = false;
    }

    void OnTriggerEnter2D(Collider2D other)
    {
        if (role == AgentRole.Predator && other.CompareTag("Prey"))
            NeuroGameManager.Instance?.OnCapture();

        if (role == AgentRole.Prey && other.CompareTag("Predator"))
            NeuroGameManager.Instance?.OnCapture();
    }
}
