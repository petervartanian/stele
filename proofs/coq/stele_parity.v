(* proofs/coq/stele_parity.v *)

From Coq Require Import Reals.
Open Scope R_scope.

Section Parity.

(* Languages are abstract; we only care about a finite set and a metric m: L -> [0,1]. *)

Variable L : Type.
Variable m : L -> R.

Hypothesis m_range : forall l : L, 0 <= m l <= 1.

(* Symmetric parity gap *)
Definition gap (a b : L) : R := Rabs (m a - m b).

(* lemma_id: L1_PARITY_GAP_BOUND *)

Lemma gap_sym :
  forall a b : L, gap a b = gap b a.
Proof.
  intros a b.
  unfold gap.
  replace (m b - m a) with (-(m a - m b)) by ring.
  rewrite Rabs_Ropp.
  reflexivity.
Qed.

Lemma gap_range :
  forall a b : L, 0 <= gap a b <= 1.
Proof.
  intros a b.
  unfold gap.
  specialize (m_range a).
  specialize (m_range b).
  intros [Ha0 Ha1] [Hb0 Hb1].
  (* m a, m b in [0,1] implies |m a - m b| <= 1 *)
  split.
  - apply Rabs_pos.
  - assert (Hdiff: -1 <= m a - m b <= 1).
    { split; lra. }
    apply Rabs_le.
    assumption.
Qed.

End Parity.
