(* proofs/coq/stele_risk.v *)

From Coq Require Import Reals.
Open Scope R_scope.

Section CompositeRisk.

Variable M : R.
Hypothesis M_pos : 0 < M.

Definition R_threat := R.
Definition R_domain := R.
Definition O_governance := R.

Definition R_composite (rt rd og : R) : R :=
  rt * rd * og.

Definition in_range (x : R) := 0 <= x <= 1.

(* lemma_id: COMPOSITE_RISK_MONOTONE *)

Lemma composite_risk_monotone_threat :
  forall rt1 rt2 rd og,
    in_range rt1 -> in_range rt2 ->
    in_range rd  -> 0 <= og <= M ->
    rt1 <= rt2 ->
    R_composite rt1 rd og <= R_composite rt2 rd og.
Proof.
  intros rt1 rt2 rd og [Hrt1_0 Hrt1_1] [Hrt2_0 Hrt2_1] [Hrd_0 Hrd_1] [Hog_0 Hog_M] Hle.
  unfold R_composite.
  (* rd * og is nonnegative *)
  assert (Hc: 0 <= rd * og).
  { apply Rmult_le_pos; assumption. }
  (* monotone in rt *)
  apply Rmult_le_compat_r with (r:=rd*og) in Hle; try assumption.
  exact Hle.
Qed.

Lemma composite_risk_range :
  forall rt rd og,
    in_range rt -> in_range rd -> 0 <= og <= M ->
    0 <= R_composite rt rd og <= M.
Proof.
  intros rt rd og [Hrt0 Hrt1] [Hrd0 Hrd1] [Hog0 HogM].
  unfold R_composite.
  split.
  - apply Rmult_le_pos; [apply Rmult_le_pos; assumption|assumption].
  - (* upper bound: rt, rd <= 1 and og <= M *)
    assert (H1: rt * rd <= 1).
    { (* 0 <= rt, rd <= 1 => product <= 1 *)
      assert (H: rt * rd <= 1 * 1) by (apply Rmult_le_compat; lra).
      simpl in H. exact H.
    }
    replace (rt * rd * og) with ((rt * rd) * og) by ring.
    eapply Rle_trans.
    + apply Rmult_le_compat_r; try assumption.
    + nra.
Qed.

End CompositeRisk.
