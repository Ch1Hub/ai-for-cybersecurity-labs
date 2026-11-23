# Security policy and responsible use

This repository is a **study archive**. It documents coursework carried out in isolated, deliberately vulnerable environments at ESI SBA. It is not a toolkit for attacking third-party systems.

## Scope and authorisation

- Every offensive technique was performed against a lab target supplied by the course: a local VM, a Docker network, a synthetic capture or a purpose-built vulnerable application.
- No real third-party system was scanned, exploited or accessed.
- The presence of a technique here is not permission to use it elsewhere. Running any of it against systems you do not own, or do not have explicit written authorisation to test, may be illegal.

## Malware handling

Live malware is **not** published in this repository.

- Executable samples (the downloader, the AgentTesla binary and its stage-2 DLL, the Mirai ELF, the Windows 11 installer payloads and extracted resources) were removed.
- What remains is analysis material: SHA-256 hashes, YARA rules, disassembly listings, FLOSS output, entropy calculations, reports and the benign C2/payload simulators written for the lab.
- The hashes are provided so the samples can be obtained from a controlled source (for example MalwareBazaar) inside an isolated analysis VM if needed.
- Never execute unknown binaries on a workstation. Use a disposable, network-isolated VM with snapshots.

## Secrets and personal data

The archive was sanitised before publication:

- Keycloak client secrets, LDAP bind credentials and the Flask session key were replaced with `REDACTED_*` placeholders.
- Wi-Fi SSIDs/passwords, SSH demo passwords and OpenStack lab passwords were redacted.
- Ganache test private keys and the mnemonic were redacted; the deterministic test accounts are worthless but are not reproduced verbatim.
- Lab private keys (`*.key`) and virtualenvs were excluded entirely.
- Operator notes, shell history, session transcripts and scripts containing a local sudo password were excluded.

If you find a secret that was missed, please open a private report through GitHub's security advisory feature or contact the maintainer directly instead of opening a public issue.

## Safe lab practice

- Run vulnerable applications only on a host-only or isolated Docker network.
- Do not expose the deliberately vulnerable containers to the public internet.
- Treat every capture, disk image and firmware dump as untrusted input.
- Keep the offensive labs (web, wireless, ethical hacking) inside a VM you can delete.

## Reporting

For a security concern about this repository itself, contact [@Ch1Hub](https://github.com/Ch1Hub) privately. Please do not test the repository's linked infrastructure; there is none.
