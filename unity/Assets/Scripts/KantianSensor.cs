using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// LEVEL 0 — Senzor Kantian.
/// Spațiu și timp sunt a priori: agentul "știe" că ceva care accelerează
/// spre el este pericol, FĂRĂ să fi învățat asta din experiență.
/// Nu se poate dezactiva, nu se poate ignora. Hardcodat.
/// </summary>
public class KantianSensor : MonoBehaviour
{
    [Header("Praguri kantiene")]
    [Tooltip("Dacă looming_index > prag → reflex de evadare instant")]
    public float loomingDangerThreshold = 0.08f;
    public float reflexSpeed = 6f;
    public float reflexDuration = 0.3f;   // secunde de override asupra Level 1

    private bool _reflexActive = false;
    private float _reflexTimer = 0f;
    private Vector2 _reflexDirection = Vector2.zero;

    // Read-only pentru QLearningAgent
    public bool ReflexActive => _reflexActive;
    public Vector2 ReflexDirection => _reflexDirection;
    public float CurrentLoomingIndex { get; private set; }

    void Update()
    {
        if (_reflexActive)
        {
            _reflexTimer -= Time.deltaTime;
            if (_reflexTimer <= 0f) _reflexActive = false;
        }

        // Scanăm toate obiectele din scenă
        float maxLoom = 0f;
        Vector2 mostDangerousDir = Vector2.zero;

        WorldObject[] objects = FindObjectsByType<WorldObject>(FindObjectsSortMode.None);
        foreach (var obj in objects)
        {
            float loom = obj.GetLoomingIndex(transform.position);
            if (loom > maxLoom)
            {
                maxLoom = loom;
                // Direcție de evadare = opusul direcției obiectului
                mostDangerousDir = ((Vector2)transform.position - (Vector2)obj.transform.position).normalized;
            }
        }

        CurrentLoomingIndex = maxLoom;

        if (maxLoom > loomingDangerThreshold)
        {
            _reflexActive = true;
            _reflexTimer = reflexDuration;
            _reflexDirection = mostDangerousDir;
        }
    }

    /// <summary>
    /// Calculează viteza de mișcare pentru acest frame.
    /// Dacă reflexul e activ, OVERRIDE complet asupra Level 1.
    /// </summary>
    public Vector2 GetVelocityOverride()
    {
        if (_reflexActive)
            return _reflexDirection * reflexSpeed;
        return Vector2.zero;
    }
}
