"""
examples/aogl_demo.py — AOGL Demo Scripti

5 senaryo:
1. Dosya tahmini (doğru)
2. Kod davranışı tahmini (doğru)
3. Yanlış tahmin (kasıtlı, yüksek confidence)
4. NO_ACTION (doğrulanamaz tahmin)
5. Mini kalibrasyon döngüsü (15-20 tahmin)
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

# Projeyi Python path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from sca.actions import ActionExecutor, CodeExecutorTool, FileReaderTool, UserAskerTool
from sca.actions import create_default_executor
from sca.aogl import AOGLController
from sca.calibration import CalibrationLearner
from sca.confidence import (
    CompositeConfidenceScorer,
    ProvenancePenaltyCalculator,
    SelfConsistencyScorer,
    VerifierScorer,
)
from sca.context import ContextBlock, ContextManager, Provenance
from sca.evaluation import OutcomeEvaluator
from sca.grounding import GroundingLog
from sca.llm import LLMClient
from sca.prediction import ActionProposal, ActionType


# ---------------------------------------------------------------------------
SEPARATOR = "=" * 60
MINI_SEPARATOR = "-" * 40


def header(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def subheader(title: str) -> None:
    print(f"\n{MINI_SEPARATOR}")
    print(f"  {title}")
    print(MINI_SEPARATOR)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------

def build_components(api_key: str, tmp_dir: str):
    """Demo için tüm AOGL bileşenlerini oluşturur."""
    llm = LLMClient(api_key=api_key)
    verifier_llm = LLMClient(api_key=api_key)
    psm = ContextManager()

    # Temel bağlam bloğu ekle
    psm.add(ContextBlock("Demo ortamı aktif.", provenance=Provenance.SYSTEM))

    sc_scorer = SelfConsistencyScorer(llm_client=llm, n_samples=1)  # Demo: tek örnek
    v_scorer = VerifierScorer(verifier_llm_client=verifier_llm)
    prov_calc = ProvenancePenaltyCalculator(context_manager=psm)

    csm = CompositeConfidenceScorer(
        self_consistency_scorer=sc_scorer,
        verifier_scorer=v_scorer,
        provenance_calculator=prov_calc,
    )

    executor = create_default_executor(
        allowed_dirs=[tmp_dir],
        sandbox=True,
        user_callback=lambda q: "Evet, kullanıcı olarak onaylıyorum.",
    )

    evaluator = OutcomeEvaluator(llm_client=llm)
    grounding_log = GroundingLog(db_path=":memory:")
    calibration_learner = CalibrationLearner(
        grounding_log=grounding_log,
        min_samples_per_category=5,  # Demo için küçük threshold
    )

    ctrl = AOGLController(
        psm_manager=psm,
        csm_scorer=csm,
        action_executor=executor,
        outcome_evaluator=evaluator,
        grounding_log=grounding_log,
        calibration_learner=calibration_learner,
        action_planner_llm=llm,
    )

    return ctrl, grounding_log, csm


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

async def scenario_1_file_prediction(ctrl: AOGLController, tmp_dir: str) -> None:
    header("Senaryo 1: Dosya Tahmini (Doğru)")

    # Sentetik test dosyası oluştur
    test_file = Path(tmp_dir) / "ornek.txt"
    first_line = "AOGL Demo — Merhaba Dünya"
    test_file.write_text(f"{first_line}\n2. satır\n3. satır\n", encoding="utf-8")
    print(f"✓ Test dosyası oluşturuldu: {test_file}")
    print(f"  Dosyanın ilk satırı: '{first_line}'")

    statement = f"'{test_file}' dosyasının ilk satırı '{first_line}' ifadesini içeriyor"
    start = time.monotonic()

    # Direkt eylem öner ve çalıştır (LLM çağrısı atlayarak hızlandır)
    pred = ctrl.make_prediction(
        statement=statement,
        category="file_location",
        confidence=0.9,
        context_block_ids=[0],
    )

    proposal = ActionProposal(
        action_type=ActionType.READ_FILE,
        parameters={"path": str(test_file)},
        expected_outcome=f"Dosya '{first_line}' içeriyor",
        cost_estimate=0.1,
        justification="Dosyayı okuyarak ilk satırı doğruluyoruz",
    )

    outcome = await ctrl.execute_and_record(pred, proposal)
    elapsed = time.monotonic() - start

    print(f"\n📋 Tahmin: {statement}")
    print(f"🔍 Eylem: READ_FILE → {test_file.name}")
    print(f"📄 Gerçek Sonuç (ilk 100 karakter): {outcome.actual_result[:100]!r}")
    print(f"✅ Match Score: {outcome.match_score:.3f}")
    print(f"💬 Gerekçe: {outcome.match_reasoning[:80]}")
    print(f"⏱  Süre: {elapsed:.2f}s")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

async def scenario_2_code_behavior(ctrl: AOGLController) -> None:
    header("Senaryo 2: Kod Davranışı Tahmini")

    code = "def f(x):\n    return x * 2\nprint(f(3))"
    statement = "f(3) fonksiyonu 6 döndürür"

    print(f"📋 Tahmin: {statement}")
    print(f"💻 Çalıştırılacak kod:\n{code}")

    pred = ctrl.make_prediction(
        statement=statement,
        category="code_behavior",
        confidence=0.95,
        context_block_ids=[0],
    )

    proposal = ActionProposal(
        action_type=ActionType.EXECUTE_CODE,
        parameters={"code": code},
        expected_outcome="6",
        cost_estimate=1.0,
        justification="Kodu çalıştırarak f(3)'ün dönüş değerini doğruluyoruz",
    )

    start = time.monotonic()
    outcome = await ctrl.execute_and_record(pred, proposal)
    elapsed = time.monotonic() - start

    print(f"\n🔍 Eylem: EXECUTE_CODE")
    print(f"📤 Çıktı: {outcome.actual_result!r}")
    print(f"✅ Match Score: {outcome.match_score:.3f}")
    print(f"💬 Gerekçe: {outcome.match_reasoning[:80]}")
    print(f"⏱  Süre: {elapsed:.2f}s")



# ---------------------------------------------------------------------------

async def scenario_3_wrong_prediction(ctrl: AOGLController) -> None:
    header("Senaryo 3: Yanlış Tahmin (Kasıtlı — Yüksek Confidence)")

    code = "def g(x):\n    return x + 1\nprint(g(5))"
    wrong_statement = "g(5) fonksiyonu 10 döndürür"  # YANLIŞ (gerçekte 6 döndürür)

    print(f"📋 Tahmin (YANLIŞ): {wrong_statement}")
    print(f"💻 Gerçek kod:\n{code}")
    print("⚠  Bu tahmin kasıtlı olarak yanlış — AOGL bunu tespit edecek")

    pred = ctrl.make_prediction(
        statement=wrong_statement,
        category="code_behavior",
        confidence=0.92,  # Yüksek confidence ama YANLIŞ
        context_block_ids=[0],
        metadata={"intentionally_wrong": True},
    )

    proposal = ActionProposal(
        action_type=ActionType.EXECUTE_CODE,
        parameters={"code": code},
        expected_outcome="10",
        cost_estimate=1.0,
        justification="Kodu çalıştırarak doğruluyoruz",
    )

    start = time.monotonic()
    outcome = await ctrl.execute_and_record(pred, proposal)
    elapsed = time.monotonic() - start

    print(f"\n🔍 Eylem: EXECUTE_CODE")
    print(f"📤 Gerçek çıktı: {outcome.actual_result!r}")
    print(f"❌ Match Score: {outcome.match_score:.3f} (düşük bekleniyor)")
    print(f"💬 Gerekçe: {outcome.match_reasoning[:80]}")
    print(f"📊 GroundingLog'da kayıtlı: 'confident but wrong' (conf=0.92, match={outcome.match_score:.2f})")
    print(f"⏱  Süre: {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Senaryo 4: NO_ACTION
# ---------------------------------------------------------------------------

async def scenario_4_no_action(ctrl: AOGLController) -> None:
    header("Senaryo 4: NO_ACTION (Doğrulanamaz Tahmin)")

    statement = "bu kullanıcı şu anda mutlu hissediyor"
    print(f"📋 Tahmin: {statement}")
    print("ℹ  Bu tahmin otomatik olarak doğrulanamaz → NO_ACTION bekleniyor")

    pred = ctrl.make_prediction(
        statement=statement,
        category="user_intent",
        confidence=0.4,
        context_block_ids=[0],
    )


    no_action_proposal = ActionProposal(
        action_type=ActionType.NO_ACTION,
        parameters={},
        expected_outcome="",
        cost_estimate=0.0,
        justification="Kullanıcının duygusal durumu otomatik araçlarla doğrulanamaz",
    )

    pred.proposed_action = no_action_proposal

    print(f"\n🔍 Eylem: NO_ACTION")
    print(f"💬 Gerekçe: {no_action_proposal.justification}")
    print(f"📊 Outcome: None (tahmin doğrulanmadı, kalibrasyon datasına katılmaz)")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

async def scenario_5_calibration(ctrl: AOGLController, grounding_log: GroundingLog, csm: CompositeConfidenceScorer) -> None:
    header("Senaryo 5: Mini Kalibrasyon Döngüsü (20 Tahmin)")

    print("Sentetik tahmin verileri oluşturuluyor...")
    print("(Bir kısmı doğru, bir kısmı yanlış — kasıtlı)")


    test_cases = [
        ("print(2+2)", "2+2 işlemi 4 verir", True, 0.85),
        ("print(3*3)", "3*3 işlemi 9 verir", True, 0.90),
        ("print(10-3)", "10-3 işlemi 7 verir", True, 0.88),
        ("print(5//2)", "5//2 işlemi 2 verir", True, 0.75),
        ("print(2**3)", "2**3 işlemi 8 verir", True, 0.80),
        ("print(2+2)", "2+2 işlemi 5 verir", False, 0.82),    # YANLIŞ
        ("print(3*3)", "3*3 işlemi 12 verir", False, 0.78),   # YANLIŞ
        ("print(10-3)", "10-3 işlemi 5 verir", False, 0.70),  # YANLIŞ
        ("print(5//2)", "5//2 işlemi 3 verir", False, 0.73),  # YANLIŞ
        ("print(len('hello'))", "len('hello') 5 verir", True, 0.91),
        ("print(len('world'))", "len('world') 5 verir", True, 0.89),
        ("print(len('ab'))", "len('ab') 2 verir", True, 0.86),
        ("print(len('abc'))", "len('abc') 5 verir", False, 0.77),  # YANLIŞ
        ("print(abs(-5))", "abs(-5) 5 verir", True, 0.93),
        ("print(abs(-5))", "abs(-5) -5 verir", False, 0.72),  # YANLIŞ
        ("x = [1,2,3]; print(len(x))", "listenin uzunluğu 3'tür", True, 0.87),
        ("x = [1,2,3]; print(len(x))", "listenin uzunluğu 4'tür", False, 0.69),  # YANLIŞ
        ("print(max(1,2,3))", "max(1,2,3) 3 verir", True, 0.94),
        ("print(min(1,2,3))", "min(1,2,3) 1 verir", True, 0.92),
        ("print(sum([1,2,3]))", "sum([1,2,3]) 6 verir", True, 0.95),
    ]

    correct_count = 0
    total_count = 0
    api_calls = 0

    for i, (code, statement, expected_correct, confidence) in enumerate(test_cases):
        pred = ctrl.make_prediction(
            statement=statement,
            category="code_behavior",
            confidence=confidence,
            context_block_ids=[0],
            metadata={"expected_correct": expected_correct, "test_idx": i},
        )

        proposal = ActionProposal(
            action_type=ActionType.EXECUTE_CODE,
            parameters={"code": code},
            expected_outcome=statement,
            cost_estimate=1.0,
            justification="Kodu çalıştırarak doğruluyoruz",
        )

        outcome = await ctrl.execute_and_record(pred, proposal)
        api_calls += 1 

        if outcome.match_score >= 0.5:
            correct_count += 1
        total_count += 1

        status = "✅" if outcome.match_score >= 0.5 else "❌"
        print(
            f"  [{i+1:2d}/20] {status} conf={confidence:.2f} "
            f"match={outcome.match_score:.2f} | {statement[:45]}"
        )

    print(f"\n📊 Sonuç: {correct_count}/{total_count} tahmin doğrulandı")
    print(f"📞 API çağrısı (evaluation): {api_calls}")


    subheader("Kalibrasyon Güncelleniyor")
    stats = await ctrl.update_calibration()
    print(f"✓ Öğrenilen kategori sayısı: {stats['categories_learned']}")
    print(f"✓ Toplam örnek: {stats['total_samples']}")
    print(f"✓ Kategoriler: {stats['categories']}")


    subheader("Kalibrasyon Raporu")
    report = ctrl.calibration_learner.report()
    print(report)


    if "code_behavior" in stats["categories"]:
        calibrators = ctrl.calibration_learner._calibrators
        cal = calibrators.get("code_behavior")
        if cal and cal.is_fitted:
            subheader("Kalibrasyon Etkisi (code_behavior kategorisi)")
            test_scores = [0.3, 0.5, 0.7, 0.85, 0.95]
            for raw in test_scores:
                calibrated = cal.apply(raw)
                print(f"  Ham: {raw:.2f}  →  Kalibre: {calibrated:.2f}")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

async def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY ortam değişkeni bulunamadı.")
        print("   export GROQ_API_KEY='gsk_...' komutunu çalıştırın.")
        sys.exit(1)

    print("\n" + "🚀 " * 10)
    print("  SCA — AOGL Demo")
    print("  Action-Outcome Grounding Loop")
    print("🚀 " * 10)

    overall_start = time.monotonic()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ctrl, grounding_log, csm = build_components(api_key, tmp_dir)
        print(f"\n✓ Bileşenler başlatıldı: tmp_dir={tmp_dir}")

        try:
            await scenario_1_file_prediction(ctrl, tmp_dir)
        except Exception as e:
            print(f"⚠  Senaryo 1 hatası: {e}")

        try:
            await scenario_2_code_behavior(ctrl)
        except Exception as e:
            print(f"⚠  Senaryo 2 hatası: {e}")

        try:
            await scenario_3_wrong_prediction(ctrl)
        except Exception as e:
            print(f"⚠  Senaryo 3 hatası: {e}")

        try:
            await scenario_4_no_action(ctrl)
        except Exception as e:
            print(f"⚠  Senaryo 4 hatası: {e}")

        try:
            await scenario_5_calibration(ctrl, grounding_log, csm)
        except Exception as e:
            print(f"⚠  Senaryo 5 hatası: {e}")

        # Genel istatistikler
        header("Genel İstatistikler")
        total_elapsed = time.monotonic() - overall_start
        all_records = grounding_log.query_all()
        with_outcomes = [(p, o) for p, o in all_records if o is not None]
        avg_match = (
            sum(o.match_score for _, o in with_outcomes) / len(with_outcomes)
            if with_outcomes
            else 0.0
        )

        print(f"⏱  Toplam süre: {total_elapsed:.1f}s")
        print(f"📊 Toplam tahmin: {len(all_records)}")
        print(f"📊 Doğrulanmış tahmin: {len(with_outcomes)}")
        print(f"📊 Ortalama match_score: {avg_match:.3f}")

        grounding_log.close()

    print("\n✅ Demo tamamlandı.\n")


if __name__ == "__main__":
    asyncio.run(main())
