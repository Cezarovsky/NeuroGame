using UnityEngine;

/// <summary>
/// Un obiect din lumea NeuroGame.
/// Poate fi: neutru, periculos (stă pe loc), sau prădător (urmărește agentul).
/// </summary>
public class WorldObject : MonoBehaviour
{
    [Header("Tip")]
    public bool isDangerous = false;
    public bool isPursuer = false;

    [Header("Mișcare")]
    public float speed = 2f;
    public float acceleration = 0f;   // prădătorul accelerează în timp

    private Transform _target;        // agentul — setat de NeuroGameManager
    private Rigidbody2D _rb;

    void Awake()
    {
        _rb = GetComponent<Rigidbody2D>();
    }

    public void SetTarget(Transform target)
    {
        _target = target;
    }

    void FixedUpdate()
    {
        if (!isPursuer || _target == null) return;

        // Accelerare progresivă — prădătorul devine mai rapid
        speed += acceleration * Time.fixedDeltaTime;

        Vector2 dir = (_target.position - transform.position).normalized;
        _rb.linearVelocity = dir * speed;
    }

    /// <summary>
    /// Looming index: cât de rapid și direct vine înspre un punct.
    /// approach_speed / distance — formula kantiană.
    /// </summary>
    public float GetLoomingIndex(Vector2 observerPos)
    {
        Vector2 toObserver = observerPos - (Vector2)transform.position;
        float distance = toObserver.magnitude;
        if (distance < 0.01f) return 999f;

        Vector2 vel = _rb != null ? _rb.linearVelocity : Vector2.zero;
        float approachSpeed = Vector2.Dot(vel, toObserver.normalized);

        return approachSpeed / distance;
    }

    void OnTriggerEnter2D(Collider2D other)
    {
        if (other.CompareTag("Agent"))
            NeuroGameManager.Instance?.OnAgentHit(isDangerous || isPursuer);
    }
}
