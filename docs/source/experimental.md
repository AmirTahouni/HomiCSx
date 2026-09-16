# Experimental features

The `homicsx.stochastic` and `homicsx.visualization` modules are retained for compatibility and evaluation, but they are not part of the publication-supported API.

Experimental features:

- may have incomplete test coverage;
- may not receive compatibility-preserving changes;
- may change or be removed before HomiCSx 1.0; and
- should not currently be relied on for unattended or long-lived research workflows.

The random geometry generators used by the supported geometry workflow are not classified as experimental merely because their algorithms are stochastic. This status applies to the higher-level ensemble, parameter-sweep, plotting, and interactive visualization conveniences.

Publication examples and validation claims do not rely on the experimental modules.

See the repository's `PUBLIC_API.md` for the compatibility policy and
{doc}`limitations` for the publication-supported feature boundary.
