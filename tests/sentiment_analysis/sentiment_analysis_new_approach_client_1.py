import time
import random
import asyncio
import sys
import os
from textblob import TextBlob
import nltk

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from developer_sdk import crowdio_connect, crowdio_map, crowdio_disconnect, CROWDio

# Download required NLTK data
nltk.download('punkt_tab')

@CROWDio.task(
    checkpoint=True,
    checkpoint_interval=3.0,  # Checkpoint every 3 seconds
    checkpoint_state=["sentiment", "confidence", "progress_percent", "processing_stage"]
)
def sentiment_worker(text):
    """
    Function to be executed on worker devices for sentiment analysis
    WITH DECLARATIVE CHECKPOINTING - PURE LOGIC, NO RESUME CODE!
    
    Args:
        text: Text segment to analyze for sentiment
    
    Returns:
        Dictionary containing sentiment analysis results
        
    Note:
        The @CROWDio.task decorator enables automatic checkpointing:
        - State variables are captured automatically via frame introspection
        - TRANSPARENT RESUME - framework handles everything automatically!
        - Just write your pure algorithm logic
        - Include 'progress_percent' in checkpoint_state for progress tracking
        
        DEVELOPER WRITES PURE LOGIC - NO RESUME CODE NEEDED!
    """
    import time
    import random
    from textblob import TextBlob

    start = time.time()
    
    # Minimum execution time to ensure checkpointing (in seconds)
    MIN_EXECUTION_TIME = 5.0  # Run for at least 5 seconds to capture checkpoints
    
    # ========================================================================
    # CHECKPOINT STATE VARIABLES - just declare them normally!
    # Framework handles resume automatically - no manual checkpoint code needed!
    # ========================================================================
    sentiment = 0.0
    confidence = 0.0
    progress_percent = 0.0
    processing_stage = "initializing"
    
    print(f"[Worker] Starting sentiment analysis (target runtime: {MIN_EXECUTION_TIME}s)")
    
    try:
        # Stage 1: Pre-processing
        processing_stage = "preprocessing"
        progress_percent = 10.0
        time.sleep(0.5)
        
        # Stage 2: Sentiment analysis
        processing_stage = "analyzing"
        progress_percent = 40.0
        analysis = TextBlob(text)
        sentiment = analysis.sentiment.polarity  # [-1, 1]
        time.sleep(1.0)
        
        # Stage 3: Confidence calculation
        processing_stage = "calculating_confidence"
        progress_percent = 70.0
        # Fake confidence: higher magnitude = higher confidence
        confidence = min(1.0, abs(sentiment) + random.uniform(0.1, 0.3))
        time.sleep(1.0)
        
        # Stage 4: Finalizing
        processing_stage = "finalizing"
        progress_percent = 90.0
        # Simulate device variability
        time.sleep(random.uniform(0.05, 0.15))
        
        progress_percent = 100.0
        processing_stage = "complete"
        
        # Ensure minimum execution time for checkpointing
        elapsed = time.time() - start
        if elapsed < MIN_EXECUTION_TIME:
            remaining = MIN_EXECUTION_TIME - elapsed
            print(f"[Worker] Waiting {remaining:.1f}s to ensure checkpoints are captured...")
            time.sleep(remaining)
        
        latency_ms = int((time.time() - start) * 1000)
        
        result = {
            "sentiment": sentiment,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "status": "success"
        }
        
        print(f"[Worker] Completed sentiment analysis | sentiment: {sentiment:.3f} | confidence: {confidence:.3f} | time: {latency_ms/1000:.1f}s")
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        latency_ms = int((time.time() - start) * 1000)
        
        return {
            "sentiment": 0.0,
            "confidence": 0.0,
            "latency_ms": latency_ms,
            "status": "error",
            "error": str(e)
        }

async def run_distributed_sentiment(text, foreman_host="localhost"):
    def split_text(text):
        blob = TextBlob(text)
        return [str(sentence) for sentence in blob.sentences]

    def aggregate_results(results):
        numerator = sum(r["confidence"] * r["sentiment"] for r in results)
        denominator = sum(r["confidence"] for r in results)
        return numerator / denominator if denominator != 0 else 0

    # Connect to foreman
    await crowdio_connect(foreman_host, 9000)

    try:
        # Split text
        segments = split_text(text)

        # Distribute sentiment analysis to workers
        worker_results = await crowdio_map(sentiment_worker, segments)

        # Parse results - handle both dict results (already parsed) and string results
        parsed_results = []
        for idx, result in enumerate(worker_results):
            if isinstance(result, dict):
                parsed_results.append(result)
            elif isinstance(result, str):
                # Try JSON first, then ast.literal_eval as fallback
                import json
                import ast
                try:
                    parsed_results.append(json.loads(result))
                except json.JSONDecodeError:
                    try:
                        parsed_results.append(ast.literal_eval(result))
                    except (ValueError, SyntaxError) as e:
                        print(f"Warning: Could not parse result {idx}: {result!r}")
                        print(f"Parse error: {e}")
                        # Use None for unparseable results
                        parsed_results.append({"sentiment": 0, "confidence": 0, "latency_ms": 0, "error": str(e)})
            else:
                parsed_results.append(result)

        # Add segment_ids to results
        results = []
        for idx, result in enumerate(parsed_results):
            result["segment_id"] = f"seg-{idx}"
            results.append(result)

        # Aggregate
        final_sentiment = aggregate_results(results)

        return final_sentiment, results

    finally:
        # Disconnect
        await crowdio_disconnect()

async def main():
    if len(sys.argv) != 2:
        print("Usage: python sentiment_analysis_new_approach_client_1.py <foreman_host>")
        print("Example: python sentiment_analysis_new_approach_client_1.py localhost")
        sys.exit(1)

    foreman_host = sys.argv[1]

    sample_text = """
    The infrastructure performed exceptionally well today.
    Network latency was minimal and the devices responded quickly.
    However, battery drain remains a concern for older phones.
    Overall, the experiment was successful and promising.
    The testing phase revealed impressive stability across all nodes.
    Data synchronization was smooth and error-free.
    Performance metrics exceeded our initial expectations significantly.
    The distributed architecture handled peak loads gracefully.
    Worker nodes maintained consistent throughput throughout the session.
    Resource utilization remained within acceptable parameters.
    The system demonstrated excellent fault tolerance capabilities.
    Response times were well below the target thresholds.
    Memory consumption stayed stable even under heavy workloads.
    The load balancing algorithm distributed tasks efficiently.
    CPU usage patterns showed optimal distribution across cores.
    Network bandwidth was utilized effectively without bottlenecks.
    The monitoring dashboard provided valuable real-time insights.
    Error rates were minimal throughout the entire test period.
    The backup systems functioned perfectly when activated.
    Data integrity checks passed without any discrepancies.
    The caching mechanism significantly improved performance.
    Garbage collection cycles had minimal impact on operations.
    Thread management was handled efficiently by the runtime.
    Database queries executed faster than previous benchmarks.
    The connection pool maintained optimal size automatically.
    Transaction rollbacks worked flawlessly when needed.
    API response times were consistently under 100 milliseconds.
    The authentication layer processed requests without delays.
    Security protocols were enforced without performance penalties.
    Logging overhead remained negligible during peak traffic.
    The message queue handled bursts effectively.
    Serialization and deserialization were optimized well.
    The framework proved to be robust and reliable.
    Configuration changes took effect immediately as expected.
    The deployment process completed without any issues.
    Rolling updates were executed seamlessly across instances.
    Health checks reported green status continuously.
    The failover mechanism activated within milliseconds.
    Data replication maintained consistency across regions.
    The microservices communicated efficiently via APIs.
    Service discovery worked perfectly in the cluster.
    Container orchestration managed resources intelligently.
    The networking layer handled traffic spikes admirably.
    SSL termination performed without introducing latency.
    Rate limiting protected services from abuse effectively.
    The circuit breaker prevented cascade failures successfully.
    Retry logic ensured eventual consistency appropriately.
    Timeout configurations were well-tuned for stability.
    The scheduler distributed jobs fairly across workers.
    Priority queues ensured critical tasks completed first.
    Deadlock detection prevented system hangs reliably.
    Resource cleanup happened automatically as designed.
    The garbage collector tuning reduced pause times.
    Memory leaks were absent in all components tested.
    The profiler identified optimization opportunities clearly.
    Performance regression tests passed all criteria.
    The continuous integration pipeline ran smoothly.
    Automated tests covered edge cases comprehensively.
    Code coverage reached acceptable levels consistently.
    Static analysis tools reported minimal warnings.
    Security scans found no critical vulnerabilities.
    Dependency updates integrated without breaking changes.
    The documentation was clear and up to date.
    API contracts remained stable across versions.
    Backward compatibility was maintained successfully.
    Migration scripts executed without data loss.
    The rollback procedure worked as documented.
    Disaster recovery drills completed successfully.
    Backup restoration took less time than expected.
    Data export formats were flexible and complete.
    Import validation caught corrupted data effectively.
    The audit trail captured all significant events.
    Compliance checks passed regulatory requirements.
    Privacy controls functioned as specified.
    Encryption protected sensitive data appropriately.
    Key rotation happened without service disruption.
    Certificate renewals were automated successfully.
    Access control policies enforced permissions correctly.
    User authentication remained secure and fast.
    Session management prevented unauthorized access.
    The notification system delivered messages promptly.
    Email templates rendered correctly across clients.
    Push notifications reached devices reliably.
    SMS delivery rates met service level agreements.
    Webhook callbacks were processed without delays.
    The event sourcing pattern proved valuable.
    CQRS separation improved read performance.
    The saga pattern handled distributed transactions.
    Eventual consistency was acceptable for use cases.
    The domain model captured business logic accurately.
    Validation rules prevented invalid state transitions.
    The repository pattern abstracted data access cleanly.
    Unit of work managed transactions effectively.
    Dependency injection simplified testing significantly.
    Mock objects facilitated isolated unit tests.
    Integration tests verified component interactions.
    End-to-end tests validated user workflows.
    Performance tests established baseline metrics.
    Stress tests identified system limits clearly.
    Load tests confirmed capacity planning estimates.
    Chaos engineering revealed resilience gaps.
    The canary deployment caught issues early.
    Blue-green deployment minimized downtime successfully.
    Feature flags enabled controlled rollouts.
    A/B testing provided valuable user insights.
    Analytics tracked user behavior comprehensively.
    Metrics dashboards visualized trends effectively.
    Alerting rules notified teams appropriately.
    Incident response procedures worked efficiently.
    Post-mortem analyses improved future reliability.
    The knowledge base grew with documented solutions.
    Team collaboration tools enhanced productivity.
    Code reviews maintained quality standards.
    Pair programming sessions shared knowledge well.
    Retrospectives identified improvement opportunities.
    Sprint planning sessions were productive.
    Stand-up meetings kept everyone synchronized.
    Technical debt was managed proactively.
    Refactoring efforts improved code maintainability.
    The architecture evolved appropriately over time.
    Design patterns were applied judiciously.
    SOLID principles guided implementation decisions.
    DRY principle reduced code duplication effectively.
    KISS philosophy kept solutions simple.
    YAGNI prevented premature optimization.
    The codebase remained readable and organized.
    Naming conventions were followed consistently.
    Comments explained complex logic helpfully.
    Type hints improved code clarity significantly.
    Linters enforced style guidelines automatically.
    Formatters maintained consistent code appearance.
    Version control history was clean and meaningful.
    Commit messages followed best practices.
    Branch strategies facilitated parallel development.
    Pull requests received timely reviews.
    Merge conflicts were resolved efficiently.
    The build process was fast and reliable.
    Artifacts were versioned and stored properly.
    Release notes communicated changes clearly.
    Changelog entries were detailed and accurate.
    Semantic versioning conveyed compatibility information.
    Deprecation warnings gave users migration time.
    Breaking changes were announced in advance.
    The upgrade path was smooth and documented.
    Customer feedback influenced roadmap priorities.
    User experience improvements drove iterations.
    Performance optimization was an ongoing effort.
    Scalability concerns were addressed proactively.
    Cost optimization reduced infrastructure expenses.
    Resource provisioning matched actual demand.
    Auto-scaling policies adjusted capacity automatically.
    Spot instances reduced compute costs significantly.
    Reserved capacity provided cost predictability.
    The multi-cloud strategy provided vendor flexibility.
    Hybrid deployment supported legacy systems.
    Edge computing reduced latency for users.
    Content delivery networks accelerated asset delivery.
    Compression reduced bandwidth consumption effectively.
    Lazy loading improved initial page load times.
    Code splitting reduced bundle sizes significantly.
    Tree shaking eliminated unused code efficiently.
    Minification reduced file sizes appropriately.
    Asset optimization improved overall performance.
    Browser caching reduced server load notably.
    Service workers enabled offline functionality.
    Progressive web app features enhanced usability.
    Mobile responsiveness worked across devices.
    Touch interactions were intuitive and smooth.
    Accessibility features supported diverse users.
    Screen readers navigated content properly.
    Keyboard navigation functioned completely.
    Color contrast met accessibility standards.
    Font sizes were readable and adjustable.
    Internationalization supported multiple languages.
    Localization adapted to regional preferences.
    Currency formatting displayed correctly everywhere.
    Date and time formats respected locale settings.
    Right-to-left languages rendered properly.
    Character encoding handled Unicode correctly.
    The search functionality returned relevant results.
    Filtering options narrowed results effectively.
    Sorting capabilities met user expectations.
    Pagination handled large datasets efficiently.
    Infinite scrolling provided smooth browsing.
    The recommendation engine suggested relevant items.
    Personalization improved user engagement.
    Machine learning models predicted accurately.
    Training pipelines processed data efficiently.
    Feature engineering improved model performance.
    Hyperparameter tuning optimized results systematically.
    Cross-validation prevented overfitting effectively.
    Model evaluation metrics guided improvements.
    A/B testing validated model improvements.
    Deployment pipelines automated model updates.
    Model monitoring detected drift proactively.
    Retraining schedules kept models current.
    Explainability features built user trust.
    Bias detection ensured fairness appropriately.
    Privacy-preserving techniques protected user data.
    Federated learning enabled collaborative training.
    Transfer learning accelerated model development.
    Ensemble methods improved prediction accuracy.
    Neural architecture search automated design.
    AutoML reduced manual tuning effort.
    The data pipeline processed terabytes daily.
    ETL jobs completed within scheduled windows.
    Data quality checks validated input integrity.
    Schema evolution handled changes gracefully.
    Partitioning improved query performance significantly.
    Indexing accelerated data retrieval operations.
    Denormalization optimized read-heavy workloads.
    Materialized views cached expensive computations.
    Query optimization reduced execution times.
    Execution plans were analyzed and tuned.
    Statistics were updated regularly automatically.
    Vacuum operations maintained database health.
    Backup strategies ensured data durability.
    Point-in-time recovery was tested regularly.
    High availability configurations prevented downtime.
    Read replicas distributed query load effectively.
    Sharding distributed data across nodes.
    Consistent hashing enabled smooth redistribution.
    The distributed consensus protocol prevented splits.
    Leader election happened quickly after failures.
    Quorum requirements ensured data safety.
    Replication lag remained acceptably low.
    Conflict resolution preserved data integrity.
    Vector clocks tracked causality accurately.
    Merkle trees verified data consistency efficiently.
    Bloom filters reduced unnecessary lookups.
    Skip lists provided efficient ordered access.
    B-trees balanced read and write performance.
    LSM trees optimized write-heavy workloads.
    Column stores compressed data effectively.
    Time-series databases handled temporal data.
    Graph databases modeled relationships naturally.
    Document stores provided schema flexibility.
    Key-value stores offered simplicity and speed.
    In-memory databases delivered ultra-low latency.
    The observability platform provided deep insights.
    Distributed tracing revealed request flows.
    Span analysis identified bottlenecks accurately.
    Correlation IDs connected related operations.
    Context propagation worked across boundaries.
    Metrics aggregation summarized performance data.
    Histogram buckets captured latency distributions.
    Percentile calculations revealed tail latencies.
    Counter metrics tracked event frequencies.
    Gauge metrics showed current system state.
    Logs provided detailed diagnostic information.
    Structured logging facilitated automated parsing.
    Log aggregation centralized troubleshooting data.
    Search capabilities found relevant log entries.
    Retention policies balanced cost and utility.
    The infrastructure as code approach worked.
    Terraform modules defined resources declaratively.
    Ansible playbooks configured systems consistently.
    CloudFormation templates managed AWS resources.
    Kubernetes manifests described desired state.
    Helm charts packaged applications effectively.
    GitOps principles automated deployments safely.
    """

    print("Distributed Sentiment Analysis Client")
    print("=" * 40)

    try:
        final_sentiment, worker_outputs = await run_distributed_sentiment(
            sample_text,
            foreman_host=foreman_host
        )

        print("=== Worker Outputs ===")
        for r in worker_outputs:
            print(r)

        print("\nFinal Aggregated Sentiment:", round(final_sentiment, 3))

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())