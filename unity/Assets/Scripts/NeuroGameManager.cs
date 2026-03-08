using UnityEngine;
using UnityEngine.UI;
using System.Collections;

/// <summary>
/// Manager adversarial: lup vs. iepure.
/// Reward-urile sunt OPUSE — suma lor e zero (zero-sum game).
/// Lup câștigă exact cât pierde iepurele.
/// </summary>
public class NeuroGameManager : MonoBehaviour
{
    public static NeuroGameManager Instance { get; private set; }

    [Header("Prefabs agenți")]
    public GameObject preyPrefab;       // iepure — alb/gri
    public GameObject predatorPrefab;   // lup — roșu

    [Header("Configurare teren")]
    [Tooltip("Teren plat, fără obstacole — prima fază")]
    public float arenaSize = 8f;

    [Header("Episoade")]
    public float episodeTimeLimit = 30f;
    public int maxEpisodes = 10000;

    [Header("Reward — Iepure")]
    public float preyRewardSurvival = 0.1f;
    public float preyPenaltyCapture = -20f;

    [Header("Reward — Lup (opus iepurelui)")]
    public float predatorRewardCapture = 20f;
    public float predatorPenaltyTime = -0.05f;

    [Header("Spawn")]
    public float minSpawnDistance = 5f;

    [Header("UI")]
    public Text episodeText;
    public Text preyRewardText;
    public Text predatorRewardText;
    public Text epsilonText;

    private GameObject _preyGO;
    private GameObject _predatorGO;
    private NeuroAgent _prey;
    private NeuroAgent _predator;

    private int _episodeCount = 0;
    private float _episodeTimer = 0f;
    private bool _episodeActive = false;
    private bool _captured = false;
    private int _captureCount = 0;
    private int _escapeCount = 0;

    void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
    }

    void Start() => StartEpisode();

    void Update()
    {
        if (!_episodeActive) return;

        _episodeTimer += Time.deltaTime;

        _prey?.ReceiveReward(preyRewardSurvival * Time.deltaTime, GetStateKey(_prey));
        _predator?.ReceiveReward(predatorPenaltyTime * Time.deltaTime, GetStateKey(_predator));

        if (_captured || _episodeTimer >= episodeTimeLimit)
            EndEpisode();

        UpdateUI();
    }

    void StartEpisode()
    {
        _episodeCount++;
        _episodeTimer = 0f;
        _captured = false;
        _episodeActive = true;

        if (_preyGO) Destroy(_preyGO);
        if (_predatorGO) Destroy(_predatorGO);

        Vector2 preyPos = RandomPosition();
        Vector2 predPos;
        int attempts = 0;
        do { predPos = RandomPosition(); attempts++; }
        while (Vector2.Distance(preyPos, predPos) < minSpawnDistance && attempts < 50);

        _preyGO = Instantiate(preyPrefab, preyPos, Quaternion.identity);
        _predatorGO = Instantiate(predatorPrefab, predPos, Quaternion.identity);

        _prey = _preyGO.GetComponent<NeuroAgent>();
        _predator = _predatorGO.GetComponent<NeuroAgent>();

        _prey.role = AgentRole.Prey;
        _prey.agentId = "prey";
        _prey.SetOpponent(_predator);

        _predator.role = AgentRole.Predator;
        _predator.agentId = "predator";
        _predator.SetOpponent(_prey);

        _prey.ResetEpisode();
        _predator.ResetEpisode();

        NeuromorphicEventBus.Instance?.OnEpisodeStart(_episodeCount);
    }

    void EndEpisode()
    {
        _episodeActive = false;

        if (_captured) _captureCount++;
        else _escapeCount++;

        NeuromorphicEventBus.Instance?.OnEpisodeEnd(new EpisodeSummary
        {
            episode = _episodeCount,
            captured = _captured,
            duration = _episodeTimer,
            preyReward = _prey?.EpisodeReward ?? 0f,
            predatorReward = _predator?.EpisodeReward ?? 0f,
            preyLoomingPeak = _prey?.CurrentLoomingIndex ?? 0f,
            predatorLoomingPeak = _predator?.CurrentLoomingIndex ?? 0f,
            captureRate = _captureCount / (float)_episodeCount,
            escapeRate = _escapeCount / (float)_episodeCount
        });

        if (_episodeCount < maxEpisodes)
            StartEpisode();
    }

    public void OnCapture()
    {
        if (!_episodeActive) return;
        _prey?.ReceiveReward(preyPenaltyCapture, "terminal");
        _predator?.ReceiveReward(predatorRewardCapture, "terminal");
        _captured = true;
    }

    Vector2 RandomPosition()
    {
        float h = arenaSize / 2f;
        return new Vector2(Random.Range(-h, h), Random.Range(-h, h));
    }

    string GetStateKey(NeuroAgent agent)
    {
        if (agent == null) return "0,0,0,0";
        Camera cam = Camera.main;
        Vector3 vp = cam.WorldToViewportPoint(agent.transform.position);
        int gx = Mathf.Clamp(Mathf.FloorToInt(vp.x * 10), 0, 9);
        int gy = Mathf.Clamp(Mathf.FloorToInt(vp.y * 10), 0, 9);
        return $"{gx},{gy},0,0";
    }

    void UpdateUI()
    {
        if (episodeText) episodeText.text = $"Ep: {_episodeCount}/{maxEpisodes}";
        if (preyRewardText) preyRewardText.text = $"Iepure: {_prey?.EpisodeReward:F1}";
        if (predatorRewardText) predatorRewardText.text = $"Lup: {_predator?.EpisodeReward:F1}";
        if (epsilonText) epsilonText.text = $"ε prey:{_prey?.Epsilon:F3} lup:{_predator?.Epsilon:F3}";
    }
}

[System.Serializable]
public struct EpisodeSummary
{
    public int episode;
    public bool captured;
    public float duration;
    public float preyReward;
    public float predatorReward;
    public float preyLoomingPeak;
    public float predatorLoomingPeak;
    public float captureRate;
    public float escapeRate;
}
