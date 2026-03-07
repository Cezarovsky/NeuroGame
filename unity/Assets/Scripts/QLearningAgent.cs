using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// LEVEL 1 — Agent Piagetian (Q-Learning tabular).
/// Agentul NU știe ce e pericol la start.
/// Descoperă prin consecințe: reward pozitiv = supraviețuire, reward negativ = coliziune.
/// Level 0 (KantianSensor) are PRIORITATE ABSOLUTĂ — dacă reflexul e activ, Level 1 tace.
/// </summary>
public class QLearningAgent : MonoBehaviour
{
    [Header("Hiperparametri")]
    public float learningRate = 0.1f;
    public float discountFactor = 0.95f;
    public float epsilon = 1.0f;          // explorare inițială completă
    public float epsilonDecay = 0.9995f;
    public float epsilonMin = 0.05f;
    public float moveSpeed = 3f;

    [Header("Discretizare spațiu")]
    public int gridSize = 10;             // 10x10 celule din ecran
    public int loomingBuckets = 4;        // 0-low, 1-medium, 2-high, 3-extreme

    // Acțiuni: 0=stay, 1=up, 2=down, 3=left, 4=right
    private const int NUM_ACTIONS = 5;
    private Dictionary<string, float[]> _qTable = new();

    private KantianSensor _kantian;
    private Rigidbody2D _rb;
    private int _currentAction = 0;
    private string _currentStateKey = "";
    private float _episodeReward = 0f;
    private int _steps = 0;

    // Stats publice pentru NeuroGameManager
    public float EpisodeReward => _episodeReward;
    public int Steps => _steps;
    public float Epsilon => epsilon;
    public int QTableSize => _qTable.Count;

    void Awake()
    {
        _kantian = GetComponent<KantianSensor>();
        _rb = GetComponent<Rigidbody2D>();
    }

    void FixedUpdate()
    {
        // Level 0 are prioritate absolută
        if (_kantian != null && _kantian.ReflexActive)
        {
            _rb.linearVelocity = _kantian.GetVelocityOverride();
            return;
        }

        // Level 1: Q-learning
        string stateKey = GetStateKey();
        _currentAction = SelectAction(stateKey);
        _rb.linearVelocity = ActionToVelocity(_currentAction);
        _currentStateKey = stateKey;
    }

    string GetStateKey()
    {
        // Normalizăm poziția în celule de grid
        Camera cam = Camera.main;
        Vector3 vp = cam.WorldToViewportPoint(transform.position);
        int gx = Mathf.Clamp(Mathf.FloorToInt(vp.x * gridSize), 0, gridSize - 1);
        int gy = Mathf.Clamp(Mathf.FloorToInt(vp.y * gridSize), 0, gridSize - 1);

        float loom = _kantian != null ? _kantian.CurrentLoomingIndex : 0f;
        int loomBucket = loom < 0.02f ? 0 : loom < 0.05f ? 1 : loom < 0.1f ? 2 : 3;
        int reflexFlag = (_kantian != null && _kantian.ReflexActive) ? 1 : 0;

        return $"{gx},{gy},{loomBucket},{reflexFlag}";
    }

    int SelectAction(string stateKey)
    {
        if (!_qTable.ContainsKey(stateKey))
            _qTable[stateKey] = new float[NUM_ACTIONS];

        if (Random.value < epsilon)
            return Random.Range(0, NUM_ACTIONS);  // explorare

        // Exploatare: alege acțiunea cu Q maxim
        float[] q = _qTable[stateKey];
        int best = 0;
        for (int i = 1; i < NUM_ACTIONS; i++)
            if (q[i] > q[best]) best = i;
        return best;
    }

    Vector2 ActionToVelocity(int action)
    {
        return action switch
        {
            1 => Vector2.up * moveSpeed,
            2 => Vector2.down * moveSpeed,
            3 => Vector2.left * moveSpeed,
            4 => Vector2.right * moveSpeed,
            _ => Vector2.zero
        };
    }

    /// <summary>
    /// Chiamat de NeuroGameManager după fiecare step cu reward-ul rezultat.
    /// </summary>
    public void ReceiveReward(float reward, string nextStateKey)
    {
        _episodeReward += reward;
        _steps++;

        if (!_qTable.ContainsKey(_currentStateKey))
            _qTable[_currentStateKey] = new float[NUM_ACTIONS];
        if (!_qTable.ContainsKey(nextStateKey))
            _qTable[nextStateKey] = new float[NUM_ACTIONS];

        // Q-update: Bellman equation
        float[] q = _qTable[_currentStateKey];
        float[] qNext = _qTable[nextStateKey];
        float maxNext = qNext[0];
        foreach (var v in qNext) if (v > maxNext) maxNext = v;

        q[_currentAction] += learningRate * (reward + discountFactor * maxNext - q[_currentAction]);

        // Epsilon decay
        epsilon = Mathf.Max(epsilonMin, epsilon * epsilonDecay);
    }

    public void ResetEpisode()
    {
        _episodeReward = 0f;
        _steps = 0;
    }
}
