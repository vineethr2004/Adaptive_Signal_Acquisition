"""Build the living technical report for the adaptive sensing project.

Update this source alongside each completed stage, then regenerate the PDF.
The report deliberately records negative results and implementation limits, so
it remains an auditable account rather than a retrospective success story.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPOSITORY_ROOT / "output" / "pdf" / "Adaptive_Signal_Acquisition_Living_Report.pdf"
STAGE_1_PLOT = REPOSITORY_ROOT / "artifacts" / "toy_run" / "toy_run.png"
STAGE_2_PLOT = REPOSITORY_ROOT / "artifacts" / "stage2_baselines" / "nmse_vs_budget.png"
STAGE_3_PLOT = REPOSITORY_ROOT / "artifacts" / "stage3_adaptive" / "nmse_vs_budget.png"
STAGE_4A_PLOT = REPOSITORY_ROOT / "artifacts" / "stage4a_diagnostics" / "nmse_gamma_ablation.png"
STAGE_4A_TRACE_PLOT = REPOSITORY_ROOT / "artifacts" / "stage4a_diagnostics" / "trace_diagnostics.png"


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def equation(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace(" ", "&nbsp;"), style)


def section(story: list[object], title: str, body: list[str], styles: dict[str, ParagraphStyle]) -> None:
    story.append(Spacer(1, 0.18 * cm))
    story.append(paragraph(title, styles["Heading2"]))
    for item in body:
        story.append(paragraph(item, styles["BodyText"]))
        story.append(Spacer(1, 0.10 * cm))


def caption(story: list[object], text: str, styles: dict[str, ParagraphStyle]) -> None:
    story.append(paragraph(text, styles["Caption"]))
    story.append(Spacer(1, 0.2 * cm))


def add_plot(story: list[object], path: Path, label: str, styles: dict[str, ParagraphStyle]) -> None:
    if path.exists():
        image = Image(str(path), width=15.6 * cm, height=9.75 * cm)
        story.extend([image, Spacer(1, 0.08 * cm)])
        caption(story, label, styles)
    else:
        story.append(paragraph(f"{label} Plot unavailable at report-build time.", styles["BodyText"]))


def footer(canvas, document) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D7DEE7"))
    canvas.line(document.leftMargin, 1.45 * cm, A4[0] - document.rightMargin, 1.45 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#556070"))
    canvas.drawString(document.leftMargin, 0.95 * cm, "Adaptive Signal Acquisition - Living Technical Report")
    canvas.drawRightString(A4[0] - document.rightMargin, 0.95 * cm, f"Page {document.page}")
    canvas.restoreState()


def build_report() -> Path:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        rightMargin=2.0 * cm,
        leftMargin=2.0 * cm,
        topMargin=1.8 * cm,
        bottomMargin=2.0 * cm,
        title="Adaptive Signal Acquisition - Living Technical Report",
        author="Adaptive Signal Acquisition project",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17365D"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#556070"),
            spaceAfter=18,
        )
    )
    styles["Heading1"].fontName = "Helvetica-Bold"
    styles["Heading1"].fontSize = 16
    styles["Heading1"].leading = 20
    styles["Heading1"].textColor = colors.HexColor("#17365D")
    styles["Heading1"].spaceAfter = 8
    styles["Heading2"].fontName = "Helvetica-Bold"
    styles["Heading2"].fontSize = 12.5
    styles["Heading2"].leading = 16
    styles["Heading2"].textColor = colors.HexColor("#1F4E79")
    styles["Heading2"].spaceBefore = 8
    styles["Heading2"].spaceAfter = 5
    styles["BodyText"].fontName = "Helvetica"
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 13.5
    styles["BodyText"].spaceAfter = 0
    styles.add(
        ParagraphStyle(
            name="Equation",
            parent=styles["BodyText"],
            fontName="Courier",
            fontSize=9.2,
            leading=14,
            leftIndent=0.8 * cm,
            textColor=colors.HexColor("#253746"),
            spaceBefore=3,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Callout",
            parent=styles["BodyText"],
            backColor=colors.HexColor("#EDF4FB"),
            borderColor=colors.HexColor("#9CC2E5"),
            borderWidth=0.6,
            borderPadding=8,
            leading=14,
            spaceBefore=5,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#556070"),
        )
    )

    story: list[object] = []
    story.append(Spacer(1, 1.5 * cm))
    story.append(paragraph("Adaptive Signal Acquisition", styles["ReportTitle"]))
    story.append(paragraph("Living Technical Report - Stages 1 through 4A", styles["ReportSubtitle"]))
    story.append(paragraph("Purpose", styles["Heading2"]))
    story.append(
        paragraph(
            "This report is the project record. It explains the mathematical model, implementation choices, controlled experiments, measured results, limitations, and next decision. It must be updated whenever a new stage changes the evidence.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(paragraph("Current evidence-based conclusion", styles["Heading2"]))
    story.append(
        paragraph(
            "Stage 4A isolated the failure in the first information-guided policy. Across 100 paired trials, large provisional-LASSO weights make selected rows more correlated and the sensing matrix less well conditioned. The Gaussian-only policy is competitive and beats random at B=32, but it does not use observation values. The next task is adaptive_v2: retain geometric diversity while introducing a better estimate of observation-dependent uncertainty.",
            styles["Callout"],
        )
    )
    summary_rows = [
        ["Stage", "Status", "Main evidence"],
        ["1", "Complete", "Deterministic simulator and visual sanity checks"],
        ["2", "Complete", "LASSO reconstruction, fixed/random paired baselines"],
        ["3", "Complete", "Sequential information policy and 100-trial comparison"],
        ["4A", "Complete", "Gamma ablation, score attribution, diversity diagnostics"],
        ["4B", "Next", "Observation-dependent adaptive_v2 and controlled validation"],
    ]
    summary_table = Table(summary_rows, colWidths=[1.3 * cm, 2.4 * cm, 11.5 * cm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.7),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7D3E0")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7FAFD")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([Spacer(1, 0.35 * cm), summary_table, PageBreak()])

    story.append(paragraph("1. Research question and model", styles["Heading1"]))
    section(
        story,
        "1.1 Project question",
        [
            "When only a small number of noisy measurements can be taken, can selecting each new measurement using earlier observations reconstruct a sparse signal more accurately than selecting all measurements in advance?",
            "The operational goal is lower normalized mean squared error (NMSE) under the same measurement budget, hidden signal distribution, candidate measurements, noise draw, and final reconstruction algorithm.",
        ],
        styles,
    )
    section(
        story,
        "1.2 Sparse hidden signal",
        [
            "The current simulator uses a directly sparse vector x in R^64. It has exactly k=4 non-zero coordinates. The active locations are sampled uniformly without replacement and their amplitudes are Gaussian with standard deviation 1.0. The estimator does not receive x or the support locations.",
            "More generally, a signal may be sparse only after a known transform Psi, written x = Psi z with ||z||_0 = k. The present implementation uses Psi = I, so z and x are the same vector.",
        ],
        styles,
    )
    story.append(equation("x in R^N,    ||x||_0 = k << N,    N = 64,    k = 4", styles["Equation"]))
    section(
        story,
        "1.3 Noisy linear measurements",
        [
            "A measurement is not a direct sample of one coordinate. It is a weighted linear combination of all 64 coordinates. A finite dictionary D contains 128 allowed normalized measurement rows. Selecting B rows from D forms the sensing matrix A in R^(B x 64).",
            "For selected action a_t, the environment returns one noisy scalar y_t. Stacking all selected actions and observations gives y = A x + epsilon.",
        ],
        styles,
    )
    story.append(equation("y_t = a_t^T x + epsilon_t,    epsilon_t ~ N(0, sigma^2),    sigma = 0.10", styles["Equation"]))
    story.append(equation("A_B = [a_1^T; a_2^T; ...; a_B^T],    y_1:B = A_B x + epsilon", styles["Equation"]))
    story.append(PageBreak())

    story.append(paragraph("2. Stage 1 - deterministic foundations", styles["Heading1"]))
    section(
        story,
        "2.1 What was built",
        [
            "Stage 1 made every object in the model concrete. A seeded random-number generator creates the sparse signal, the 128-by-64 candidate dictionary, a fixed list of 16 selected actions, and a Gaussian noise vector. Repeating the seed reproduces the same arrays exactly.",
            "The initial example used x with shape (64,), the candidate dictionary with shape (128, 64), A with shape (16, 64), and y with shape (16,). At this stage there was no reconstruction and no adaptive selection.",
        ],
        styles,
    )
    add_plot(story, STAGE_1_PLOT, "Figure 1. Stage 1 visual sanity check: sparse signal, clean measurements, noise, and observations.", styles)
    section(
        story,
        "2.2 Why deterministic simulation matters",
        [
            "A stochastic experiment is not trustworthy if a result cannot be reproduced. Fixed seeds let us distinguish a code change from normal randomness. They also ensure a later adaptive method can be compared against the same hidden signal and the same noise realization.",
        ],
        styles,
    )
    story.append(PageBreak())

    story.append(paragraph("3. Stage 2 - reconstruction and non-adaptive reference", styles["Heading1"]))
    section(
        story,
        "3.1 Final reconstruction with LASSO",
        [
            "After B measurements, LASSO reconstructs a candidate signal z. The first term rewards agreement with observations; the second term encourages most coordinates of z to become exactly zero. In this project z is an estimate of x because the signal is directly sparse.",
            "The solver is ISTA. Each iteration moves z toward lower data mismatch and then applies soft-thresholding, which shrinks weak coefficients to zero. The final LASSO configuration uses lambda=0.08, maximum 5000 iterations, and tolerance 1e-7.",
        ],
        styles,
    )
    story.append(equation("x_hat = arg min_z [ 0.5 ||A z - y||_2^2 + lambda ||z||_1 ]", styles["Equation"]))
    section(
        story,
        "3.2 Fixed and random baselines",
        [
            "Fixed sensing uses one predetermined permutation of the 128 candidate rows and takes its first B rows in every trial. Random sensing draws a new random row order before a trial begins, then takes its first B rows. Neither method looks at y before deciding its rows.",
            "For each trial and budget, fixed and random receive the identical hidden x and the identical pre-drawn noise prefix. They differ only in selected A rows. This paired design is essential for a fair comparison.",
        ],
        styles,
    )
    section(
        story,
        "3.3 Evaluation metrics",
        [
            "NMSE is the main reconstruction metric. Lower is better. The 1e-12 guard prevents division by zero in an all-zero reference. Support F1 measures whether the method identifies non-zero coordinates, using an estimate magnitude threshold of 1e-3.",
        ],
        styles,
    )
    story.append(equation("NMSE = ||x - x_hat||_2^2 / (||x||_2^2 + 1e-12)", styles["Equation"]))
    add_plot(story, STAGE_2_PLOT, "Figure 2. Stage 2 100-trial baseline result: error decreases plausibly as budget increases.", styles)
    story.append(PageBreak())

    story.append(paragraph("4. Stage 3 - sequential information-guided sensing", styles["Heading1"]))
    section(
        story,
        "4.1 Sequential policy",
        [
            "Unlike the two Stage 2 methods, Stage 3 chooses one action at a time. Before selecting action t+1, it has history h_t = (A_t, y_1:t): all earlier actions and observed values. It does not have the true hidden signal x.",
            "The selected action is appended to A; earlier rows are never changed. The simulator produces its observation only after the policy has selected that row.",
        ],
        styles,
    )
    story.append(equation("a_(t+1) = pi(h_t),    h_t = (A_t, y_1:t)", styles["Equation"]))
    section(
        story,
        "4.2 Gaussian uncertainty approximation",
        [
            "The policy begins with mean mu_0 = 0 and covariance Sigma_0 = 0.0625 I. The prior variance 0.0625 is a simple heuristic based on the current signal generator: expected active fraction k/N = 4/64 with active amplitude variance near 1.",
            "For a possible unused row a, a^T Sigma a estimates uncertainty in the measurement direction. Relative to noise variance sigma^2, the policy scores it using the linear-Gaussian mutual-information expression below and takes the largest score.",
        ],
        styles,
    )
    story.append(equation("score(a) = log(1 + (a^T Sigma a) / sigma^2)", styles["Equation"]))
    section(
        story,
        "4.3 Update after observing y_t",
        [
            "Once a row is measured, a standard Gaussian conditioning update changes the mean and covariance. K_t is the gain: it is larger in directions that were uncertain and smaller when measurement noise is large. The covariance update reduces uncertainty in the direction that was just measured.",
        ],
        styles,
    )
    story.append(equation("K_t = Sigma_t a_t / (a_t^T Sigma_t a_t + sigma^2)", styles["Equation"]))
    story.append(equation("mu_(t+1) = mu_t + K_t (y_t - a_t^T mu_t)", styles["Equation"]))
    story.append(equation("Sigma_(t+1) = Sigma_t - K_t (a_t^T Sigma_t)", styles["Equation"]))
    story.append(PageBreak())

    story.append(paragraph("5. Why the Stage 3 policy uses a support-aware proxy", styles["Heading1"]))
    section(
        story,
        "5.1 The important limitation",
        [
            "In a pure linear-Gaussian model, the covariance update depends on which rows were selected, but not directly on the numerical values observed. That is not sufficiently responsive to sparse evidence. A large or small y value should affect what we consider worth measuring next.",
            "Therefore the current implementation computes a provisional LASSO estimate z_hat_t after each observation using only A_t and y_1:t. This is not the final reconstruction and it is not an exact sparse Bayesian posterior. It is a transparent heuristic for making the score data-dependent.",
        ],
        styles,
    )
    story.append(equation("z_hat_t = arg min_z [ 0.5 ||A_t z - y_1:t||_2^2 + lambda ||z||_1 ]", styles["Equation"]))
    section(
        story,
        "5.2 Support-aware score",
        [
            "The policy adds a positive semidefinite term gamma z_hat_t z_hat_t^T to the Gaussian covariance. The default gamma is 1. Rows aligned with the provisional sparse estimate receive an increased score. Importantly, this estimate arises only from already observed measurements; the action-selection function has no hidden_signal or x argument.",
        ],
        styles,
    )
    story.append(equation("Sigma_tilde = Sigma_t + gamma z_hat_t z_hat_t^T", styles["Equation"]))
    story.append(equation("a_(t+1) = arg max over unused a of log(1 + (a^T Sigma_tilde a)/sigma^2)", styles["Equation"]))
    section(
        story,
        "5.3 What is recorded",
        [
            "For every adaptive step, the code writes budget, trial, step, selected candidate index, observed value, and selected information score. These action traces make the policy inspectable rather than a black box.",
        ],
        styles,
    )
    story.append(PageBreak())

    story.append(paragraph("6. Stage 3 controlled result", styles["Heading1"]))
    section(
        story,
        "6.1 Protocol",
        [
            "The primary Stage 3 run used 100 independent trials, N=64, k=4, 128 candidate rows, noise standard deviation 0.1, budgets B = 8, 12, 16, 20, 24, 32, and the same final LASSO for every method. Fixed, random, and information-guided sensing used matched hidden signals and matched noise prefixes within each trial and budget.",
            "The table reports mean NMSE. Lower is better. Values are from artifacts/stage3_adaptive/adaptive_summary.csv with seed 20260913.",
        ],
        styles,
    )
    result_rows = [
        ["B", "Fixed", "Random", "Information-guided", "Best baseline"],
        ["8", "0.8324", "0.8238", "0.9006", "Random"],
        ["12", "0.5616", "0.6010", "0.7795", "Fixed"],
        ["16", "0.4045", "0.4185", "0.6499", "Fixed"],
        ["20", "0.3024", "0.3396", "0.5078", "Fixed"],
        ["24", "0.2400", "0.2792", "0.4028", "Fixed"],
        ["32", "0.1815", "0.2004", "0.2724", "Fixed"],
    ]
    result_table = Table(result_rows, colWidths=[1.2 * cm, 2.3 * cm, 2.3 * cm, 4.2 * cm, 3.3 * cm])
    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (3, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7D3E0")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7FAFD")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([Spacer(1, 0.2 * cm), result_table, Spacer(1, 0.3 * cm)])
    add_plot(story, STAGE_3_PLOT, "Figure 3. Stage 3 100-trial result: the first information-guided policy has higher mean NMSE than the best baseline at every tested budget.", styles)
    # Keep the interpretation together instead of leaving a single orphaned line after the plot.
    story.append(PageBreak())
    section(
        story,
        "6.2 Interpretation",
        [
            "The result answers the current question for this exact policy and setting: the first information-guided proxy does not improve final sparse reconstruction. At B=32, its mean NMSE is 0.2724, compared with 0.1815 for fixed sensing. This is a meaningful gap, not just a visual tie.",
            "The result does not prove that all adaptive sensing is worse. It says that this particular greedy Gaussian-plus-provisional-LASSO policy is not better in this direct-sparse, dense-random-dictionary regime.",
        ],
        styles,
    )

    story.append(paragraph("7. Stage 4A - score attribution and gamma ablation", styles["Heading1"]))
    section(
        story,
        "7.1 Separating the two score components",
        [
            "Stage 3 formed one matrix Sigma_tilde = Sigma_t + gamma x_hat_t x_hat_t^T and saved only the final nonlinear information score. Stage 4A preserves that selection rule but separately records the two raw contributions for every selected action.",
            "The Gaussian component g(a) = a^T Sigma_t a measures remaining posterior variance along a. It changes when measurement directions are selected, but its covariance update does not depend on the numerical y values. The LASSO component l(a) = gamma (a^T x_hat_t)^2 measures alignment with the current provisional reconstruction and therefore depends on observed y values. The total score applies the logarithm only after adding these terms.",
        ],
        styles,
    )
    story.append(equation("g(a) = a^T Sigma_t a", styles["Equation"]))
    story.append(equation("l(a) = gamma (a^T x_hat_t)^2", styles["Equation"]))
    story.append(equation("score(a) = log(1 + (g(a) + l(a)) / sigma^2)", styles["Equation"]))
    section(
        story,
        "7.2 Experimental protocol",
        [
            "The ablation used 100 paired trials with N=64, k=4, 128 dense Rademacher candidate actions, noise standard deviation 0.1, budgets 8, 12, 16, 20, 24, and 32, final LASSO lambda 0.08, and seed 20260919. The tested gamma values were 0, 0.1, 0.5, and 1.0. Paired NMSE differences against random sensing use 2000 deterministic bootstrap resamples for 95% confidence intervals.",
            "Gamma=0 disables the LASSO component. It remains sequential because Sigma changes after selected rows, but it is not observation-value adaptive: two trials with the same dictionary select the same row sequence even if their y values differ.",
        ],
        styles,
    )
    stage4_rows = [
        ["B", "Random", "gamma=0", "gamma=0.1", "gamma=0.5", "gamma=1"],
        ["8", "0.8263", "0.7685", "0.8197", "0.8643", "0.8441"],
        ["12", "0.5876", "0.5649", "0.6188", "0.7497", "0.7033"],
        ["16", "0.3750", "0.3788", "0.4489", "0.6126", "0.5589"],
        ["20", "0.2843", "0.2607", "0.3310", "0.4868", "0.4528"],
        ["24", "0.2251", "0.2070", "0.2473", "0.3988", "0.3669"],
        ["32", "0.1663", "0.1443", "0.1633", "0.2681", "0.2312"],
    ]
    stage4_table = Table(stage4_rows, colWidths=[1.0 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm, 2.4 * cm, 2.2 * cm])
    stage4_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7D3E0")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7FAFD")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([Spacer(1, 0.15 * cm), stage4_table, Spacer(1, 0.2 * cm)])
    add_plot(story, STAGE_4A_PLOT, "Figure 4. Stage 4A mean NMSE: larger LASSO-proxy weights systematically degrade reconstruction after the smallest budget.", styles)
    story.append(PageBreak())
    story.append(paragraph("7.3 Action-level diagnostic evidence", styles["Heading2"]))
    section(
        story,
        "Observed mechanism",
        [
            "Across all adaptive steps, the mean LASSO share of the raw score component rises from 0 at gamma=0 to 0.10, 0.30, and 0.42 for gamma values 0.1, 0.5, and 1.0. Mean maximum correlation with an earlier action rises from 0.17 to 0.22, while mean sensing-matrix condition number rises from 1.79 to 2.52.",
            "Mean provisional LASSO NMSE is 0.47 for gamma=0, 0.51 for gamma=0.1, 0.62 for gamma=0.5, and 0.59 for gamma=1. The observation-dependent proxy is therefore not merely adding information: at larger weights it steers acquisition toward more redundant geometry and poorer provisional reconstructions.",
            "At B=32, gamma=0 improves paired mean NMSE over random by -0.0220 with 95% bootstrap interval [-0.0424, -0.0011]. Gamma=1 is worse by +0.0649 with interval [+0.0277, +0.1007]. Most gamma=0 results at smaller budgets have intervals crossing zero, so they are described as competitive rather than conclusively superior.",
        ],
        styles,
    )
    if STAGE_4A_TRACE_PLOT.exists():
        story.append(Image(str(STAGE_4A_TRACE_PLOT), width=15.6 * cm, height=11.35 * cm))
        caption(story, "Figure 5. Stage 4A action diagnostics: proxy dominance, row correlation, condition number, and provisional reconstruction error.", styles)
    story.append(PageBreak())

    story.append(paragraph("8. Diagnosis and Stage 4B decision", styles["Heading1"]))
    section(
        story,
        "8.1 What Stage 4A established",
        [
            "The Gaussian component is useful for geometrically balanced experimental design. It reduces variance in directions not yet well measured and tends to preserve a better-conditioned sensing matrix. However, because its covariance update ignores observed values, gamma=0 is not the observation-guided solution sought by the project.",
            "The current LASSO outer-product component does use observations, but it treats large estimated signal magnitude as if it were uncertainty. Those are different concepts. The term gamma (a^T x_hat)^2 rewards rows aligned with what the estimator already believes, and large gamma can overpower the diversity supplied by the Gaussian covariance.",
        ],
        styles,
    )
    section(
        story,
        "8.2 Correct next move: adaptive_v2",
        [
            "Freeze gamma=1 as adaptive_v1 and retain gamma=0 as a non-observation-dependent sequential-design reference. Build adaptive_v2 with an initial random or Gaussian-diverse burn-in, then estimate uncertainty from variation across multiple plausible sparse reconstructions rather than from x_hat x_hat^T magnitude.",
            "A practical next model is bootstrap-LASSO covariance: perturb or resample the observed data, reconstruct several plausible signals, calculate their empirical covariance, and score a candidate by a^T Sigma_boot a. This asks where plausible reconstructions disagree. An explicit row-diversity penalty may be added only if the covariance score does not sufficiently prevent redundancy.",
            "After adaptive_v2 is stable on the base setting, continue Stage 4 with controlled sweeps over SNR, sparsity, signal family, and dictionary family. Use 100 trials per condition and 500 for selected final figures.",
        ],
        styles,
    )
    story.append(
        paragraph(
            "Decision: Stage 4A is complete. The failure is localized to the current observation-dependent LASSO proxy, not to the entire idea of sequential experimental design. Stage 4B should replace magnitude alignment with estimated sparse-reconstruction uncertainty.",
            styles["Callout"],
        )
    )
    story.append(PageBreak())

    story.append(paragraph("9. Reproducibility and maintenance", styles["Heading1"]))
    section(
        story,
        "9.1 Repository map",
        [
            "src/adaptive_signal_acquisition/signals.py defines sparse-signal generation. measurements.py defines the candidate dictionary and y = A x + noise. reconstruction.py contains ISTA LASSO. baselines.py contains fixed/random action selection. sequential.py contains score attribution and the sequential loop. experiments.py creates Stage 2/3 paired studies. stage4.py creates the controlled gamma ablation and bootstrap intervals.",
            "scripts/run_baseline_study.py creates Stage 2 artifacts. scripts/run_adaptive_study.py creates Stage 3 artifacts. scripts/run_stage4a_diagnostics.py creates Stage 4A artifacts. scripts/build_living_report.py regenerates this PDF.",
        ],
        styles,
    )
    section(
        story,
        "9.2 Update rule for this report",
        [
            "After every completed experiment family, update: (1) the exact configuration and seed, (2) equations or policy changes, (3) result tables and figures, (4) the interpretation including failures, and (5) the next decision. Never replace an old result without preserving the method name and configuration that produced it.",
            "The report should remain evidence-led. A conclusion must identify the relevant artifact folder, sample count, metric, and comparison protocol. This is especially important before adding an LLM research orchestrator in later stages.",
        ],
        styles,
    )
    section(
        story,
        "9.3 Commands",
        [
            "Run tests: .venv\\Scripts\\python.exe -m pytest",
            "Run Stage 2: .venv\\Scripts\\python.exe scripts\\run_baseline_study.py",
            "Run Stage 3: .venv\\Scripts\\python.exe scripts\\run_adaptive_study.py",
            "Run Stage 4A: .venv\\Scripts\\python.exe scripts\\run_stage4a_diagnostics.py",
            "Regenerate this report: use the bundled runtime or an environment with reportlab installed to run scripts\\build_living_report.py",
        ],
        styles,
    )
    story.append(Spacer(1, 0.5 * cm))
    story.append(paragraph("Report version: v2 - Stages 1 through 4A complete", styles["Caption"]))
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(build_report())
