(* proofs/coq/stele_ensembles.v *)

From Coq Require Import Reals.
From Coq Require Import List.
Import ListNotations.

Open Scope R_scope.

Section EnsembleFN.

Variable k : nat.
Hypothesis k_pos : (k > 0)%nat.

(* We model false-negative probabilities as a list [p1; ...; pk] in [0,1]. *)

Variable ps : list R.
Hypothesis ps_length : length ps = k.
Hypothesis ps_range : forall p, In p ps -> 0 <= p <= 1.

(* Abstract probability space: we only reason about the conditional probability values. *)

(* FN_or is the false-negative rate of the OR-ensemble *)
Definition FN_or : R :=
  fold_right Rmult 1 ps.

(* Lemma: if all individual FN rates are <= p, then the ensemble FN is <= p^k. *)

(* lemma_id: ENSEMBLE_OR_FN_BOUND *)
Lemma ensemble_or_fn_bound :
  forall p,
    0 <= p <= 1 ->
    (forall pi, In pi ps -> pi <= p) ->
    FN_or <= p ^ k.
  
  Lemma ensemble_or_fn_bound :
  forall p,
    0 <= p <= 1 ->
    (forall pi, In pi ps -> pi <= p) ->
    FN_or <= p ^ k.
Proof.
  intros p Hp_range Hle.
  unfold FN_or.
  (* We can prove this by induction on the list ps. *)
  assert (Hk : length ps = k) by apply ps_length.
  revert k Hk.
  induction ps as [|a ps' IH]; intros k' Hlen; simpl in *.
  - inversion Hlen. simpl. (* empty case should not occur since k > 0, but we handle it anyway *)
    rewrite Rpow_O. lra.
  - destruct k' as [|k'']; simpl in Hlen; try discriminate.
    inversion Hlen as [Hk''].
    subst.
    assert (Ha_in: In a (a :: ps')) by (left; reflexivity).
    specialize (Hle a Ha_in).
    specialize (IH k'' Hk'').
    destruct Hp_range as [Hp0 Hp1].
    (* All ps' elements are also <= p *)
    assert (Hle_tail: forall pi : R, In pi ps' -> pi <= p).
    { intros pi Hin.
      apply Hle.
      right; assumption.
    }
    specialize (IH Hle_tail).
    (* Now we use monotonicity of multiplication for nonnegative reals *)
    assert (Ha_range: 0 <= a <= 1) by (apply ps_range; left; reflexivity).
    destruct Ha_range as [Ha0 Ha1].
    assert (Hp_nonneg: 0 <= p) by exact Hp0.
    assert (Ha_le_p: a <= p) by exact Hle.
    (* 0 <= fold_right Rmult 1 ps' and <= 1 by repeated application of the range hypothesis *)
    assert (Hprod_range: 0 <= fold_right Rmult 1 ps' <= 1).
    { clear IH Hle Hle_tail Hk'' k_pos.
      (* Sketch: product of numbers in [0,1] stays in [0,1] *)
      induction ps' as [|b ps'' IHps'']; simpl.
      - split; lra.
      - assert (Hb_range: 0 <= b <= 1) by (apply ps_range; right; left; reflexivity).
        destruct Hb_range as [Hb0 Hb1].
        destruct IHps'' as [Hprod0 Hprod1].
        split.
        + apply Rmult_le_pos; assumption.
        + apply Rmult_le_1; try assumption; lra.
    }
    destruct Hprod_range as [Hprod0 Hprod1].
    (* Now use a <= p and monotonicity: a * prod <= p * prod *)
    assert (Hstep1: a * fold_right Rmult 1 ps' <= p * fold_right Rmult 1 ps').
    { apply Rmult_le_compat_r; try assumption. }
    (* And p * prod <= p^(S k'') by IH *)
    assert (Hstep2: p * fold_right Rmult 1 ps' <= p * p ^ k'').
    { apply Rmult_le_compat_l; try assumption. }
    eapply Rle_trans; [exact Hstep1|].
    eapply Rle_trans; [exact Hstep2|].
    (* p * p^k'' = p^(S k'') *)
    rewrite <- Rpow_plus.
    + simpl. replace (k'' + 1)%nat with (S k'') by lia.
      apply Req_le. reflexivity.
    + assumption.
Qed.

End EnsembleFN.
