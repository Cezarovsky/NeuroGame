using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;

/// <summary>
/// Manager principal: spawn obiecte, reward, episoade, UI stats.
/// Singleton — accesibil din orice script via NeuroGameManager.Instance
/// </summary>
public class NeuroGameManager : MonoBehaviour
{
    public static NeuroGameManager Instance { get; private set; }

    [Header("Prefabs")]
    public GameObject agentPrefab;
    public GameObject neutralObjectPrefab;
    public GameObject dangerousObjectPrefab;
    public GameObject pursuerPrefab;

    [Header("Configurare scenă")]
    public int neutralCount = 5;
    public int dangerousCount = 3;
    public int pursuerCount = 1;

    [Header("Episoade")]
    public float episodeTimeLimit = 30f;
    public int maxEpisodes = 10000;

    [Header("Reward")]
    public float rewardSurvival = 0.1f;      // fiecare secundă supraviețuită
    public float penaltyHit = -10f;

    [Header("UI")]
    public Text episodeText;
    public Text rewardText;
    public Text epsilonText;
    public Text qTableText;

    private GameObject _agentGO;
    private QLearningAgent _agent;
    private KantianSensor _kantian;
    private List<GameObject> _spawnedObjects = new();

    private int _episodeCount = 0;
    private float _episodeTimer = 0f;
    private bool _episodeActive = false;
    private bool _agentHit = false;

    // Bounds din Camera
    private float _xMin, _xMax, _yMin, _yMax;

    void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
    }

    void Start()
    {
        Camera cam = Camera.main;
        float camH = cam.orthographicSize;
        float camW = camH * cam.aspect;
        _xMin = -camW + 0.5f;
        _xMax = camW - 0.5f;
        _yMin = -camH + 0.5f;
        _yMax = camH - 0.5f;

        StartEpisode();
    }

    void Update()
    {
        if (!_episodeActive) return;

        _episodeTimer += Time.deltaTime;

        // Reward de supraviețuire per frame
        if (_agent != null)
        {
            string nextState = GetAgentStateKey();
            _agent.ReceiveReward(rewardSurvival * Time.deltaTime, nextState);
        }

        // Condiții de terminare episod
        if (_agentHit || _episodeTimer >= episodeTimeLimit)
            EndEpisode();

        UpdateUI();
    }

    string GetAgentStateKey()
    {
        if (_agent == null) return "0,0,0,0";
        Camera cam = Camera.main;
        Vector3 vp = cam.WorldToViewportPoint(_agentGO.transform.position);
        int gx = Mathf.Clamp(Mathf.FloorToInt(vp.x * 10), 0, 9);
        int gy = Mathf.Clamp(Mathf.FloorToInt(vp.y * 10), 0, 9);
        float loom = _kantian != null ? _kantian.CurrentLoomingIndex : 0f;
        int lb = loom < 0.02f ? 0 : loom < 0.05f ? 1 : loom < 0.1f ? 2 : 3;
        return $"{gx},{gy},{lb},0";
    }

    void StartEpisode()
    {
        _episodeCount++;
        _episodeTimer = 0f;
        _agentHit = false;
        _episodeActive = true;

        // Curățăm scena precedentă
        foreach (var go in _spawnedObjects) if (go) Destroy(go);
        _spawnedObjects.Clear();
        if (_agentGO) Destroy(_agentGO);

        // Spawn agent în centru
        _agentGO = Instantiate(agentPrefab, Vector3.zero, Quaternion.identity);
        _agent = _agentGO.GetComponent<QLearningAgent>();
        _kantian = _agentGO.GetComponent<KantianSensor>();
        _agent?.ResetEpisode();

        // Spawn obiecte
        SpawnObjects(neutralObjectPrefab, neutralCount, false, false);
        SpawnObjects(dangerousObjectPrefab, dangerousCount, true, false);
        SpawnObjects(pursuerPrefab, pursuerCount, true, true);
    }

    void SpawnObjects(GameObject prefab, int count, bool dangerous, bool pursuer)
    {
        for (int i = 0; i < count; i++)
        {
            // Spawn la margini, nu pe agent
            Vector2 pos = RandomBorderPosition();
            GameObject go = Instantiate(prefab, pos, Quaternion.identity);
            WorldObject wo = go.GetComponent<WorldObject>();
            if (wo != null)
            {
                wo.isDangerous = dangerous;
                wo.isPursuer = pursuer;
                if (pursuer) wo.SetTarget(_agentGO.transform);
            }
            _spawnedObjects.Add(go);
        }
    }

    Vector2 RandomBorderPosition()
    {
        // Poziție aleatorie pe una din cele 4 margini
        int edge = Random.Range(0, 4);
        return edge switch
        {
            0 => new Vector2(Random.Range(_xMin, _xMax), _yMax),   // sus
            1 => new Vector2(Random.Range(_xMin, _xMax), _yMin),   // jos
            2 => new Vector2(_xMin, Random.Range(_yMin, _yMax)),   // stânga
            _ => new Vector2(_xMax, Random.Range(_yMin, _yMax))    // dreapta
        };
    }

    void EndEpisode()
    {
        _episodeActive = false;

        if (_episodeCount < maxEpisodes)
            StartEpisode();
        else
            Debug.Log($"Training complet după {maxEpisodes} episoade.");
    }

    public void OnAgentHit(bool isDangerous)
    {
        if (!_episodeActive) return;
        if (isDangerous)
        {
            _agent?.ReceiveReward(penaltyHit, "0,0,0,0");
            _agentHit = true;
        }
    }

    void UpdateUI()
    {
        if (episodeText) episodeText.text = $"Episod: {_episodeCount}";
        if (rewardText) rewardText.text = $"Reward: {_agent?.EpisodeReward:F1}";
        if (epsilonText) epsilonText.text = $"ε: {_agent?.Epsilon:F3}";
        if (qTableText) qTableText.text = $"Q-states: {_agent?.QTableSize}";
    }
}
